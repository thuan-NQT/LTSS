#include <stdio.h>
#include <stdint.h>
#define FILTER_WIDTH 3
volatile __device__ int d_min = 0;

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};


void readPnm(char * fileName, 
	int &numChannels, int &width, int &height, uint8_t * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	if (strcmp(type, "P2") == 0)
		numChannels = 1;
	else if (strcmp(type, "P3") == 0)
		numChannels = 3;
	else // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);

	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uint8_t *)malloc(width * height * numChannels);
	for (int i = 0; i < width * height * numChannels; i++)
		fscanf(f, "%hhu", &pixels[i]);

	fclose(f);
}

void writePnm(uint8_t * pixels, int numChannels, int width, int height, 
	char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	if (numChannels == 1)
		fprintf(f, "P2\n");
	else if (numChannels == 3)
		fprintf(f, "P3\n");
	else
	{
		fclose(f);
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height * numChannels; i++)
		fprintf(f, "%hhu\n", pixels[i]);

	fclose(f);
}

//HW
__global__ void convertRgb2GrayKernel(uint8_t * inPixels, int width, int height, 
	uint8_t * outPixels)
{
	// TODO
	// Reminder: gray = 0.299*red + 0.587*green + 0.114*blue  
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;

	if(x < width && y < height)
	{
		int i = y * width + x;
		uint8_t red = inPixels[3 * i];
		uint8_t green = inPixels[3 * i + 1];
		uint8_t blue = inPixels[3 * i + 2];
		outPixels[i] = 0.299f*red + 0.587f*green + 0.114f*blue;
	}
}

//HW3
__global__ void edgeDetectKernel(uint8_t * inPixels, int width, int height, 
	int * filterX, int* filterY, int filterWidth, 
	int * outPixels)
{
	int c = threadIdx.x + blockIdx.x * blockDim.x;
	int r = threadIdx.y + blockIdx.y * blockDim.y;
  	if (r < height && c < width) 
 	{
		int i = r * width + c;
		float outx = 0.0f;
		float outy = 0.0f;
		int iFilter = 0; 
		for (int x = -1; x <= 1; x++)
			for (int y = -1; y <= 1; y++) 
			{
				int rx = r + x;
				int cy = c + y;
				if (rx < 0) rx = 0;
				if (rx > height - 1) rx = height - 1;
				if (cy < 0) cy = 0;
				if (cy > width - 1) cy = width - 1; 
				int k = rx * width + cy;
				outx += float(float(inPixels[k]) * float(filterX[iFilter]));
				outy += float(float(inPixels[k]) * float(filterY[iFilter]));

				iFilter++;
			}
    outPixels[i] = abs(outx) + abs(outy);
  	}
	//mỗi thread sẽ tính độ quan trọng của một pixel
}

__global__ void calSeamEnergyKernel(int * inPixels, int width, int height,
				int * nextPixels)
{
	
	int i = threadIdx.x + threadIdx.y*blockDim.x + gridDim.x*blockIdx.x;
	int min;

	//mỗi thread xử lí mỗi điểm ảnh từ dưới lên để tính độ quan trọng của seam
	if(i < width)
	{	
		for(int r = height-2; r >= 0; r--)
		{
			min = i + (r+1) * width;
			
			if(i-1 >= 0 && inPixels[i-1 + (r+1) * width] < inPixels[min])
				min = i-1 + (r+1) * width;

			if(i+1 <= width-1 && inPixels[i+1 + (r+1) * width] < inPixels[min])
				min = i+1 + (r+1) * width;

			inPixels[i + r * width] += inPixels[min];
			nextPixels[i + r * width] = min;
			__syncthreads();
		}
	}
}

// duyệt mảng seam để tìm ra seam nhỏ nhất
__global__ void findSeamKernel(int * energy, int * seam, int * nextPixels, int width, int height)
{
	int idx = threadIdx.x + threadIdx.y*blockDim.x + gridDim.x*blockIdx.x;
	
	if(idx < width && energy[idx] < energy[d_min])
		d_min = idx;

	__syncthreads();

	if(idx < height)
	{
		if(idx == 0)
			seam[0] = d_min;
		else
			seam[idx] = nextPixels[idx-1];
	}
}

// xóa seam và dán lại khoảng đã xóa
__global__ void removeSeamKernel(uint8_t * inPixels, int width, int height,
										int * seam, uint8_t * outPixels)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

	if(j < width && i < height)
		for(int idx = 0; idx < height-1; idx++)
		{
			if(j + i * width + idx + 1 > width*height - 1) 
				break;

			if(j + i * width >= seam[idx] - idx && j + i * width < seam[idx+1] - idx - 1)
			{
				outPixels[3*(j + i * width)] = inPixels[3*(j + i * width + idx + 1)];
				outPixels[3*(j + i * width)+1] = inPixels[3*(j + i * width + idx + 1)+1];
				outPixels[3*(j + i * width)+2] = inPixels[3*(j + i * width + idx + 1)+2];
			}
		}
}

void seamCarvingKernel(uint8_t * inPixels, int width, int height, uint8_t * outPixels,
			int * filterX, int * filterY, int filterWidth, dim3 blockSize=dim3(1, 1),
        	int times = 1)
{
	// TODO
	uint8_t * d_inGrayscale, * inGrayscale, * d_inPixels, * d_outPixels;
	uint8_t * temp = (uint8_t*)malloc(width*height*3*sizeof(uint8_t));
	memcpy(temp, inPixels, width*height*3*sizeof(uint8_t));
	int * energy = (int*)malloc(width*height*sizeof(int));
	int * min_seam = (int*)malloc(height*sizeof(int));
	int * next_pixels = (int*)malloc(width*(height-1)*sizeof(int));
	int * d_energy, *d_next_pixels, *d_filterX, *d_filterY, *d_seam;

	inGrayscale = (uint8_t*)malloc(width*height*sizeof(uint8_t));
	CHECK(cudaMalloc(&d_inPixels, width*height*3*sizeof(uint8_t)));
	CHECK(cudaMalloc(&d_outPixels, width*height*3*sizeof(uint8_t)));
	CHECK(cudaMalloc(&d_inGrayscale, width*height*sizeof(uint8_t)));
	CHECK(cudaMalloc(&d_energy, width*height*sizeof(int)));
	CHECK(cudaMalloc(&d_next_pixels, width*(height-1)*sizeof(int)));
	CHECK(cudaMalloc(&d_seam, height*sizeof(int)));
	CHECK(cudaMalloc(&d_filterX, filterWidth*filterWidth*sizeof(int)));	
	CHECK(cudaMalloc(&d_filterY, filterWidth*filterWidth*sizeof(int)));	

	dim3 gridSize((width-1)/blockSize.x + 1, (height-1)/blockSize.y + 1);
	dim3 gridSize_flatten((width-1)/(blockSize.x*blockSize.y) + 1);

	CHECK(cudaMemcpy(d_filterX, filterX, filterWidth*filterWidth*sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_filterY, filterY, filterWidth*filterWidth*sizeof(int), cudaMemcpyHostToDevice));
	

	for(int count = 0; count < times; count++)
	{
		CHECK(cudaMemcpy(d_inPixels, temp, width*height*3*sizeof(uint8_t), cudaMemcpyHostToDevice));
		convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_inGrayscale);
		cudaDeviceSynchronize();
		CHECK(cudaGetLastError());

		CHECK(cudaMemcpy(inGrayscale, d_inGrayscale, width*height*sizeof(uint8_t), cudaMemcpyDeviceToHost));

		// writePnm(inGrayscale, 1, width, height, "d_grayscale.pnm");
	
		size_t smem = (blockSize.x+filterWidth/2*2)*(blockSize.y+filterWidth/2*2)*sizeof(uint8_t);

		//Calculate energy of each pixel using Edge Detection
		edgeDetectKernel<<<gridSize, blockSize, smem>>>(d_inGrayscale, width, height, d_filterX, d_filterY, filterWidth, d_energy);
		cudaDeviceSynchronize();
		CHECK(cudaGetLastError());

		CHECK(cudaMemcpy(energy, d_energy, width*height*sizeof(int), cudaMemcpyDeviceToHost));
		// writeTxt(energy, width, height, "v2_device_pixel_energy.txt");

		//Calculate seam from pixels energy
		calSeamEnergyKernel<<<gridSize_flatten, blockSize>>>(d_energy, width, height, d_next_pixels);
		cudaDeviceSynchronize();
		CHECK(cudaGetLastError());

		CHECK(cudaMemcpy(energy, d_energy, width*height*sizeof(int), cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(next_pixels, d_next_pixels, width*(height-1)*sizeof(int), cudaMemcpyDeviceToHost));
		// writeTxt(energy, width, height, "v2_device_seam_energy.txt");

		//tìm seam nhỏ nhất(phần này cài tuần tự)
		int min = 0;
		for(int c = 1; c < width; c++)
			if(energy[c] < energy[min])
				min = c; 
	
		min_seam[0] = min;
		for(int r = 0; r < height-1; r++)
		{
			min_seam[r+1] = next_pixels[min];
			min = min_seam[r+1];
		}

		//Test seam finding
		for(int i = 0; i < height; i++)
		{
			temp[3*min_seam[i]] = 255;
			temp[3*min_seam[i] + 1] = 1;
			temp[3*min_seam[i] + 2] = 1;
		}
		// writePnm(temp, 3, width, height, "v2_device_seam.pnm");		

		CHECK(cudaMemcpy(d_seam, min_seam, height*sizeof(int), cudaMemcpyHostToDevice));

		removeSeamKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_seam, d_outPixels);
		cudaDeviceSynchronize(); 
		CHECK(cudaGetLastError());
		width--;

		//Remove min seam from the image
		CHECK(cudaMemcpy(outPixels, d_outPixels, width*height*3*sizeof(uint8_t), cudaMemcpyDeviceToHost));

		temp = outPixels;
	}
	writePnm(outPixels, 3, width, height, "seamcarving_parallel.pnm");
}

void seamCarving(uint8_t * inPixels, int width, int height, int * filterX, int* filterY, int filterWidth,
        uint8_t * outPixels, int times=1,
        bool useDevice=false, dim3 blockSize=dim3(1, 1))
{
	GpuTimer timer; 
	timer.Start();
	seamCarvingKernel(inPixels, width, height, outPixels, filterX, filterY, filterWidth, blockSize, times);
	timer.Stop();
	float time2 = timer.Elapsed();
	printf("%f ms\n", time2);
}

void printDeviceInfo()
{
	cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);
    printf("****************************\n");

}

int main(int argc, char ** argv)
{
	if (argc == 4 && argc > 6)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}

	printDeviceInfo();

	// Read input image file
	int numChannels, width, height, times;
	uint8_t * inPixels;
	dim3 blockSize(32, 32);					

	readPnm(argv[1], numChannels, width, height, inPixels);
	char* type = argv[2];
	if (argc > 3)
	{
		blockSize.x = atoi(argv[3]);
		blockSize.y = atoi(argv[4]);
	}
	if (argc == 6)
		times = atoi(argv[5]);

	if (numChannels != 3)
	{
		return EXIT_FAILURE; // Input image must be RGB
	}
	printf("\nImage size (width x height): %i x %i\n", width, height);

	// Set up a simple filter with blurring effect 
	int filterWidth = FILTER_WIDTH;
	int filterX[] = {1, 0, -1,
					2, 0, -2,
					1, 0, -1};

	int filterY[] = {1, 2, 1, 
					0, 0, 0,
					-1, -2, -1};

	// Blur input image not using device
	uint8_t * correctOutPixels = (uint8_t *)malloc(width * height * numChannels * sizeof(uint8_t)); 
	uint8_t * outPixels = (uint8_t*)malloc(width * height * numChannels * sizeof(uint8_t));

	//Seam carving by Device
	printf("Kernel time: \n");
	seamCarving(inPixels, width, height, filterX, filterY, filterWidth, outPixels, times, true, blockSize);

    // Free memories
	free(inPixels);
	free(correctOutPixels);
	free(outPixels);
}
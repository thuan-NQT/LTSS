#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>

#define FILTER_WIDTH 3
const int x_sobel_filter[FILTER_WIDTH*FILTER_WIDTH] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
const int y_sobel_filter[FILTER_WIDTH*FILTER_WIDTH] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
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

//BT01
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

//BT01
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

// TODO--------------------------------------------------

// Convolution
void convolution(uint8_t *inPixels, int width, int height, int *outPixels, const int *filter) 
{
    for (int outPixelsR = 0; outPixelsR < height; outPixelsR++)
		{
			for (int outPixelsC = 0; outPixelsC < width; outPixelsC++)
			{
				int outPixel = 0;
				for (int filterR = 0; filterR < FILTER_WIDTH; filterR++)
				{
					for (int filterC = 0; filterC < FILTER_WIDTH; filterC++)
					{
						int filterVal = filter[filterR*FILTER_WIDTH + filterC];
						int inPixelsR = outPixelsR - FILTER_WIDTH/2 + filterR;
						int inPixelsC = outPixelsC - FILTER_WIDTH/2 + filterC;
						inPixelsR = min(max(0, inPixelsR), height - 1);
						inPixelsC = min(max(0, inPixelsC), width - 1);
						uint8_t inPixel = inPixels[inPixelsR*width + inPixelsC];
						outPixel += filterVal * inPixel;
					}
				}
				outPixels[outPixelsR*width + outPixelsC] = outPixel; 
			}
		}
}

// Tìm độ quan trọng của mỗi pixel (edge detection)
void edgeDetection(uint8_t *inPixels, int width, int height, uint *importancePixels) 
{
    // Chuyển ảnh RGB sang ảnh grayscale: gray = 0.299*red + 0.587*green + 0.114*blue 
    uint8_t *grayPixels = (uint8_t*)malloc(width*height*sizeof(uint8_t));
    for (int i = 0; i < height; i++) 
    {
        for (int j = 0; j < width; j++)
        {
            int idx = width*i + j;
            grayPixels[idx] = 0.299f*inPixels[3*idx] + 0.587f*inPixels[3*idx + 1] + 0.114f*inPixels[3*idx + 2];
        }
    }

    // Phát hiện cạnh theo chiều x: Convolution với bộ lọc x-sobel
    int *edgePixels_x = (int*)malloc(width*height*sizeof(int));
    convolution(grayPixels, width, height, edgePixels_x, x_sobel_filter);

    // Phát hiện cạnh theo chiều y: Convolution với bộ lọc y-sobel
    int *edgePixels_y = (int*)malloc(width*height*sizeof(int));
    convolution(grayPixels, width, height, edgePixels_y, y_sobel_filter);

    // Tính độ quan trọng của một pixel
    for (int i = 0; i < width*height; i++)
        importancePixels[i] = abs(edgePixels_x[i]) + abs(edgePixels_y[i]);

    // Giải phóng vùng nhớ
    free(grayPixels);
    free(edgePixels_x);
    free(edgePixels_y);
}

// Tìm seam ít quan trọng nhất
void findLeastImportantSeam(uint *importancePixels, int width, int height, int *seam)
{
    uint *importanceOfSeams = importancePixels;
    int *trace = (int*)malloc(width*(height - 1)*sizeof(int)); // bỏ trace của dòng cuối

    // Tính độ quan trọng ít nhất tính tới dưới cùng
    for (int r = height - 1; r >= 0; r--)
    {
        for (int c = 0; c < width; c++)
        {
            // Đoạn này có cách nào code đẹp hơn ko-------
            int left = INT_MAX;
            int right = INT_MAX;
            int down = importanceOfSeams[width*(r + 1) + c]; 
            if (c - 1 >= 0 && c + 1 < width)
            {
                left = importanceOfSeams[width*(r + 1) + c - 1];
                right = importanceOfSeams[width*(r + 1) + c + 1];
            }
            else if (c - 1 < 0)
                right = importanceOfSeams[width*(r + 1) + c + 1];
            else
                left = importanceOfSeams[width*(r + 1) + c - 1];

            int minNextElem = min(min(left, down), right);
            importanceOfSeams[width*r + c] += minNextElem;
            if (minNextElem == down)
                trace[width*r + c] = 0;
            else if (minNextElem == left)
                trace[width*r + c] = -1;
            else
                trace[width*r + c] = 1;
            //----------------------------------------------
        }
    }

    // Truy vết và tìm seam ít quan trọng nhất
    seam[0] = 0;
    for (int i = 1; i < width; i++)
        if (importanceOfSeams[seam[0]] > importanceOfSeams[i])
            seam[0] = i;
    for (int i = 1; i < height; i++)
        seam[i] = seam[i - 1] + width + trace[seam[i - 1]];

    // Free vùng nhớ
    free(trace);
}

// Xóa seam
void removeSeam(uint8_t *pixels, int width, int height, int *seam)
{
    for (int i = 0; i < height - 1; i++)
    {
        for (int j = 3*(seam[i] - i); j < 3*seam[i + 1]; j++)
        {
            pixels[j] = pixels[j + 3*(i + 1)];
        }
    }

    for (int j = 3*(seam[height - 1] - height + 1); j < 3*(width*height - height); j++)
        pixels[j] = pixels[j + 3*height];
}

// Seam carving by host
void seamCarvingByHost(uint8_t *inPixels, int width, int height, uint8_t *outPixels, int newWidth)
{
    uint *importancePixels = (uint*)malloc(width*height*sizeof(uint));
    int *seam = (int*)malloc(height*sizeof(int));

    memcpy(outPixels, inPixels, 3*width*height*sizeof(uint8_t));

    while (width > newWidth)
    {
        importancePixels = (uint*)realloc(importancePixels, width*height*sizeof(uint));

        // Tìm độ quan trọng của mỗi pixel (edge detection)
        edgeDetection(outPixels, width, height, importancePixels);

        // Tìm seam ít quan trọng nhất
        findLeastImportantSeam(importancePixels, width, height, seam);

        // for test
        // for (int i = 0; i < height; i++)
        //     printf("%d\t", seam[i] - width*i);
        // printf("\n");

        // Xóa seam này
        removeSeam(outPixels, width, height, seam);
        width -= 1;

        // for test
        //width = 0;
    }

    // Giải phóng vùng nhớ
    free(importancePixels);
    free(seam);
}

// Seam Carving
void seamCarving(uint8_t *inPixels, int width, int height,
    uint8_t* outPixels, int newWidth,
    bool useDevice=false, dim3 blockSize=dim3(1)) 
{
    GpuTimer timer;
    timer.Start();

    if (useDevice == false)
    {
        printf("Seam Carving by host: \n");
        seamCarvingByHost(inPixels, width, height, outPixels, newWidth);
    }
    else // use device
    {
        //printf(("Seam Carving by device: \n"));
        //seamCarvingByDevice(inPixels, width, height, outPixels, newWidth, blockSize);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

char * concatStr(const char * s1, const char * s2)
{
	char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}

int main(int argc, char ** argv)
{
    // Print out device info
    printDeviceInfo();

	// Read input RGB image file
	int numChannels, width, height;
	uint8_t * inPixels;
	readPnm(argv[2], numChannels, width, height, inPixels);
	if (numChannels != 3)
		return EXIT_FAILURE; // Input image must be RGB
	printf("Image size (width x height): %i x %i\n\n", width, height);

    // Seam Carving not using device
    int newWidth = atoi(argv[1]);
	uint8_t * correctOutPixels= (uint8_t *)malloc(3 * width * height);
	seamCarving(inPixels, width, height, correctOutPixels, newWidth);

	// Seam Carving using device

	// Compute mean absolute error between host result and device result

	// Write results to files
	char * outFileNameBase = strtok(argv[3], "."); // Get rid of extension
	writePnm(correctOutPixels, 3, newWidth, height, concatStr(outFileNameBase, "_host.pnm"));
	//writePnm(outPixels, 3, width, height, concatStr(outFileNameBase, "_device.pnm"));

	// Free memories
	free(inPixels);
	free(correctOutPixels);
    
    return EXIT_SUCCESS;
}
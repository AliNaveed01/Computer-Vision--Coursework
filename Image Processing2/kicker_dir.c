#include <stdio.h>
#include <string.h>
#include <sys/unistd.h>
#include <sys/stat.h>
#include "esp_err.h"
#include "esp_log.h"
#include "esp_spiffs.h"
#include "esp_vfs_fat.h"
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include "esp_timer.h"
#include <stdlib.h>
#include <string.h>

static const char *TAG = "Kicker_spiffs";

typedef struct PGMImage {
    char pgmType[3];
    unsigned int width;
    unsigned int height;
    unsigned int maxValue;
    float *data;
} PGMImage;

void ignoreComments(FILE *fp) {
    int ch;
    char line[100];
    while ((ch = fgetc(fp)) != EOF && isspace(ch))
        ;
    if (ch == '#') {
        fgets(line, sizeof(line), fp);
        ignoreComments(fp);
    } else {
        fseek(fp, -1, SEEK_CUR);
    }
}

void openPGM(PGMImage *pgm, const char *filename) {
    FILE *pgmfile = fopen(filename, "rb");
    if (pgmfile == NULL) {
        ESP_LOGE(TAG, "File %s does not exist or can't be opened", filename);
        return;
    }

    ignoreComments(pgmfile);
    fscanf(pgmfile, "%s", pgm->pgmType);
    if (strcmp(pgm->pgmType, "P5")) {
        ESP_LOGE(TAG, "Wrong file type!");
        fclose(pgmfile);
        return;
    }

    ignoreComments(pgmfile);
    fscanf(pgmfile, "%d %d", &(pgm->width), &(pgm->height));
    ignoreComments(pgmfile);
    fscanf(pgmfile, "%d", &(pgm->maxValue));
    ignoreComments(pgmfile);

    size_t numPixels = (size_t)(pgm->width) * (pgm->height);
    pgm->data = (float *)malloc(numPixels * sizeof(float));

    if (pgm->maxValue < 256) {
        unsigned char val;
        float *ptr = pgm->data;
        for (size_t i = 0; i < numPixels; i++) {
            fread(&val, sizeof(unsigned char), 1, pgmfile);
            *ptr++ = ((float)val) / ((float)(pgm->maxValue));
        }
    } else {
        unsigned short val;
        float *ptr = pgm->data;
        for (size_t i = 0; i < numPixels; i++) {
            fread(&val, sizeof(unsigned short), 1, pgmfile);
            *ptr++ = ((float)val) / ((float)(pgm->maxValue));
        }
    }
    fclose(pgmfile);
}

int countWhitePixels(const PGMImage *img, int startX, int startY, int width,
                     int height) {
    int whitePixels = 0;
    for (int y = startY; y < startY + height; ++y) {
        for (int x = startX; x < startX + width; ++x) {
            if (img->data[y * img->width + x] > 0.8) {
                whitePixels++;
            }
        }
    }
    return whitePixels;
}

bool detect_kicker_by_white_pixels(PGMImage *img, double threshold_percentage) {
    int width = img->width;
    int height = img->height;
    int upperHeight = height / 2;
    int totalPixels = width * upperHeight;
    int whitePixels = countWhitePixels(img, 0, 0, width, upperHeight);
    double whitePixelPercentage = (whitePixels / (double)totalPixels) * 100;
    ESP_LOGI(TAG, "White Pixel Percentage in Upper Region: %.2f%%",
             whitePixelPercentage);
    return (whitePixelPercentage > threshold_percentage);
}

float calculate_difference(const PGMImage *test_roi,
                           const PGMImage *reference_roi) {
    float difference = 0.0;
    for (int y = 0; y < test_roi->height; ++y) {
        for (int x = 0; x < test_roi->width; ++x) {
            int index = y * test_roi->width + x;
            float test_pixel = test_roi->data[index];
            float reference_pixel = reference_roi->data[index];
            difference += fabs(test_pixel - reference_pixel);
        }
    }
    return difference;
}

int compare_image_rois(PGMImage *test_roi, const char *reference_filenames[],
                       int num_references) {
    float min_difference = FLT_MAX;
    int closest_match_index = -1;

    for (int i = 0; i < num_references; i++) {
        ESP_LOGI(TAG, "Loading reference image: %s", reference_filenames[i]);

        PGMImage *reference_image = malloc(sizeof(PGMImage));
        openPGM(reference_image, reference_filenames[i]);

        if (reference_image->data != NULL) {
            float difference = calculate_difference(test_roi, reference_image);
            if (difference < min_difference) {
                min_difference = difference;
                closest_match_index = i;
            }

            free(reference_image->data);
            free(reference_image);
        } else {
            ESP_LOGE(TAG, "Failed to load reference image: %s",
                     reference_filenames[i]);
        }
    }
    return closest_match_index;
}

// Resize
void resizeImage(PGMImage *pgm, unsigned int newWidth, unsigned int newHeight) {
    float *newData = (float *)malloc(newWidth * newHeight * sizeof(float));
    float scaleX = (float)(pgm->width) / (float)(newWidth);
    float scaleY = (float)(pgm->height) / (float)(newHeight);

    for (unsigned int i = 0; i < newHeight; i++) {
        for (unsigned int j = 0; j < newWidth; j++) {
            float x = j * scaleX;
            float y = i * scaleY;
            unsigned int x1 = (unsigned int)floor(x);
            unsigned int y1 = (unsigned int)floor(y);
            unsigned int x2 = x1 + 1;
            unsigned int y2 = y1 + 1;
            float dx = x - x1;
            float dy = y - y1;

            if (x2 >= pgm->width) x2 = x1;
            if (y2 >= pgm->height) y2 = y1;

            unsigned int index1 = y1 * pgm->width + x1;
            unsigned int index2 = y1 * pgm->width + x2;
            unsigned int index3 = y2 * pgm->width + x1;
            unsigned int index4 = y2 * pgm->width + x2;

            float val1 = pgm->data[index1];
            float val2 = pgm->data[index2];
            float val3 = pgm->data[index3];
            float val4 = pgm->data[index4];

            float interpolatedValue = val1 * (1 - dx) * (1 - dy) +
                                      val2 * dx * (1 - dy) +
                                      val3 * (1 - dx) * dy +
                                      val4 * dx * dy;

            unsigned int newIndex = i * newWidth + j;
            newData[newIndex] = interpolatedValue;
        }
    }

    pgm->width = newWidth;
    pgm->height = newHeight;
    free(pgm->data);
    pgm->data = newData;
}

// Helper function to log elapsed time
void log_elapsed_time(const char *stage, int64_t start_time) {
    int64_t end_time = esp_timer_get_time();
    ESP_LOGI(TAG, "%s time: %lld ms", stage, (end_time - start_time) / 1000);
}

void app_main(void) {
	//Directory File size
	int files = 8; //Change this to according to the number of files in the directory.
    ESP_LOGI(TAG, "Initializing SPIFFS");
    int64_t pipeline_start_time = 0;

    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/spiffs",
        .partition_label = NULL,
        .max_files = 8,
        .format_if_mount_failed = true
    };

    esp_err_t ret = esp_vfs_spiffs_register(&conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize SPIFFS");
        return;
    }

    for (int file_index = 1; file_index <= files; file_index++) { 
        // Generate the filename based on the current index
        pipeline_start_time = esp_timer_get_time();
        char image_path[40];
        snprintf(image_path, sizeof(image_path), "/spiffs/images/%d.pgm", file_index);

        int64_t load_start_time = esp_timer_get_time();
        PGMImage *test_image = malloc(sizeof(PGMImage));
        openPGM(test_image, image_path);
        if (test_image->data == NULL) {
            // File does not exist or could not be loaded, skip to next iteration
            free(test_image);
            ESP_LOGI(TAG, "File %s not found. Skipping to next file.", image_path);
            continue;
        }
        log_elapsed_time("Image loading", load_start_time);

        ESP_LOGI(TAG, "%s Image Loaded", image_path);

        int64_t resize_start_time = esp_timer_get_time();
        resizeImage(test_image, 128, 128);
        log_elapsed_time("Image resizing", resize_start_time);

        float threshold_percentage = 2.0;
        int64_t detection_start_time = esp_timer_get_time();
        if (detect_kicker_by_white_pixels(test_image, threshold_percentage)) {
            log_elapsed_time("Kicker detection", detection_start_time);

            const char *reference_filenames[] = {
                "/spiffs/kkrIVc.pgm", "/spiffs/kkrVc.pgm",
                "/spiffs/kkrVn.pgm", "/spiffs/kkrIVn.pgm"
            };

            int64_t match_start_time = esp_timer_get_time();
            int closest_match = compare_image_rois(test_image, reference_filenames, 4);
            log_elapsed_time("Image matching", match_start_time);

            const char *match_status = (closest_match == 0 || closest_match == 3) ? "invalid" : "valid";
            ESP_LOGI(TAG, "Kicker position is %s, closest match to reference image: %s",
                     match_status, reference_filenames[closest_match]);
        } else {
            ESP_LOGI(TAG, "Kicker not detected.");
        }

        // Free the memory for this image
        free(test_image->data);
        free(test_image);
        log_elapsed_time("Total pipeline", pipeline_start_time);
    }

    esp_vfs_spiffs_unregister(conf.partition_label);
   
}


// BEN 2026-11-01
// Change aim: Recover a known sub-pixel shift between two images with phase correlation.
// File: subpixel_registration.cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace cv;

// BEN 2026-11-01
// Change aim: Build a textured grayscale test image when the user gives no image file.
// File: subpixel_registration.cpp
// Function: makeTestImage
static Mat makeTestImage(int rows, int cols)
{
    RNG rng(12345);
    Mat noise(rows, cols, CV_32F);
    rng.fill(noise, RNG::UNIFORM, 0.0, 1.0);
    GaussianBlur(noise, noise, Size(0, 0), 3.0);   // smooth random texture
    normalize(noise, noise, 0.0, 1.0, NORM_MINMAX);
    return noise;
}

// BEN 2026-11-01
// Change aim: Translate an image by a sub-pixel amount using bicubic interpolation.
// File: subpixel_registration.cpp
// Function: shiftImage
static Mat shiftImage(const Mat& src, double dx, double dy)
{
    Mat M = (Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
    Mat dst;
    warpAffine(src, dst, M, src.size(), INTER_CUBIC, BORDER_REFLECT101);
    return dst;
}

// BEN 2026-11-01
// Change aim: Estimate the shift of image b relative to image a at sub-pixel precision.
// File: subpixel_registration.cpp
// Function: estimateShift
static Point2d estimateShift(const Mat& a, const Mat& b, double& response)
{
    Mat window;
    createHanningWindow(window, a.size(), CV_32F);   // reduces edge effects
    return phaseCorrelate(a, b, window, &response);
}

// BEN 2026-11-01
// Change aim: Run the demo, print the error and save the result images.
// File: subpixel_registration.cpp
// Function: main
int main(int argc, char** argv)
{
    // optional: pass a grayscale image path as the first argument
    Mat img;
    if (argc > 1) {
        img = imread(argv[1], IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Cannot read image: " << argv[1] << std::endl;
            return 1;
        }
        img.convertTo(img, CV_32F, 1.0 / 255.0);
    } else {
        img = makeTestImage(512, 512);
    }

    const double trueDx = 3.37, trueDy = -2.81;      // known shift (x, y) in pixels
    Mat moved = shiftImage(img, trueDx, trueDy);

    double response = 0.0;
    Point2d est = estimateShift(img, moved, response);

    // register: move the second image back by the estimated shift
    Mat corrected = shiftImage(moved, -est.x, -est.y);

    // error is measured on the inner area to ignore border fill
    Rect inner(20, 20, img.cols - 40, img.rows - 40);
    double meanAbsErr = mean(abs(img(inner) - corrected(inner)))[0];

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "true shift     (x, y): " << trueDx << ", " << trueDy << std::endl;
    std::cout << "estimated shift(x, y): " << est.x << ", " << est.y << std::endl;
    std::cout << "error          (x, y): " << std::abs(est.x - trueDx) << ", "
              << std::abs(est.y - trueDy) << " px" << std::endl;
    std::cout << "peak response: " << response << std::endl;
    std::cout << "mean abs difference after registration: " << meanAbsErr << std::endl;

    Mat out;
    hconcat(std::vector<Mat>{img, moved, corrected}, out);
    out.convertTo(out, CV_8U, 255.0);
    imwrite("subpixel_result.png", out);
    std::cout << "saved subpixel_result.png (original | shifted | corrected)" << std::endl;
    return 0;
}

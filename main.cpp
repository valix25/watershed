/* external */
#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <vector>
#include <queue>
#include <map>
#include <limits.h>
#include <unistd.h>
#include <math.h>
#include <type_traits>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
/* internal */
#include "include/lodepng.h"

#define mask -2 // initial value of a threshold value
#define wshed 0 // value of pixels belonging to watershed
#define init -1 // initial value of fo
#define inqueue -3 // value assigned to pixels put into queue

typedef unsigned int uint;
typedef unsigned char uchar;
typedef std::vector<std::vector<unsigned int> > ImageContainer;
typedef std::map<std::pair<unsigned int, unsigned int>, 
	std::vector<std::pair<unsigned int, unsigned int> > > Neighbors;
typedef std::array<std::vector<std::pair<uint, uint> >, 256 > HeightTable; 

void showHelp(char *s)
{
	std::cout << "Usage: " << s << " [-option] [argument]" << "\n";
	std::cout << "option: " << "-h show help information" << "\n";
	std::cout << "        " << "-i input image" << "\n";
	std::cout << "        " << "-o output image" << "\n";
	std::cout << "        " << "-s visualize image" << "\n";
	std::cout << "example: " << s << " -i coins.jpg -o out.jpg" << "\n";
}

const std::string currentDateTime() {
	time_t     now = time(0);
	struct tm  tstruct;
	char       buf[80];
	tstruct = *localtime(&now);
	strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
	return buf;
}

void greyscaleToBinary(ImageContainer& image, const unsigned int &cols,
	const unsigned int &rows)
{
	for(unsigned int row = 0; row < rows; row++)
		for(unsigned int col = 0; col < cols; col++)
		{
			if(image[row][col] > 127) image[row][col] = 1;
			else image[row][col] = 0;
		}
}

void decodePNG(ImageContainer& image, unsigned int &cols,
	unsigned int &rows, std::string filepath)
{
	std::vector<unsigned char> imageRaw;
	unsigned int width, height;
	unsigned error = lodepng::decode(imageRaw, width, height, filepath, LCT_GREY);

	if(error)
	{
		std::cerr << "decoder error " << error << ": ";
		std::cerr << lodepng_error_text(error) << "\n";
		exit(1); 
	}

	cols = width;
	rows = height;

	image.resize(rows);
	for(unsigned int i = 0; i < rows; i++)
	{
		image[i].resize(cols);
	}

	for(unsigned int i = 0; i < rows; i++)
		for(unsigned int j = 0; j < cols; j++)
		{
			image[i][j] = imageRaw[i*cols + j];
		}
}

void encodePNG(ImageContainer& image, unsigned int &cols,
	unsigned int &rows, std::string filepath)
{
	std::vector<unsigned char> imageRaw;
	for(unsigned int i = 0; i < rows; i++)
		for(unsigned int j = 0; j < cols; j++)
		{
			imageRaw.push_back(image[i][j]);
		}

	unsigned error = lodepng::encode(filepath, imageRaw, cols, rows, LCT_GREY);
	if(error)
	{
		std::cerr << "encoder error " << error << ": ";
		std::cerr << lodepng_error_text(error) << "\n";
	}
}

cv::Mat readImage(std::string filepath)
{
	cv::Mat image;
	image = cv::imread(filepath, CV_LOAD_IMAGE_GRAYSCALE);
	if(!image.data)
	{
		std::cerr << "Could not open or find image: " << filepath << "\n";
		exit(1);
	}
	/* convert to grayscale*/
	/*cv::Mat gray_image(image.rows, image.cols, CV_8UC1);
	cv::cvtColor(image, gray_image, CV_BGR2GRAY);
	for(uint i = 0; i < (uint) gray_image.rows; i++)
		for(uint j = 0; j < (uint) gray_image.cols; j++)
		{
			if(gray_image.at<uchar>(i,j) < 0)
			{
				std::cerr << "Negative values in the gray image conversion!\n";
				exit(1);
			}
		}*/
	return image;
}

void writeImage(const std::string& filepath, const cv::Mat& image)
{
	bool check = cv::imwrite(filepath.c_str(), image);
	if(!check)
	{
		std::cerr << "Could not write from image or to path: " << filepath << "\n";
		exit(1);
	}
	std::cout << "End write\n";
}

Neighbors computeNeighbors(const cv::Mat& image)
{
	// 4 neighbor approach: up, right, bottom, left
	Neighbors neighbors;
	for(int i = 0; i < image.rows; i++)
		for(int j = 0; j < image.cols; j++)
		{
			std::pair<uint, uint > p((uint) i, (uint) j);
			std::vector<std::pair<uint, uint> > n;
			// counterclockwise from upper neighbor
			if(i-1 >= 0)
			{
				n.push_back(std::make_pair((uint) i-1,(uint) j));
			}
			if(j+1 < image.cols)
			{
				n.push_back(std::make_pair((uint)i, (uint)j+1));
			}
			if(i+1 < image.rows)
			{
				n.push_back(std::make_pair((uint)i+1,(uint) j));
			}
			if(j-1 >= 0)
			{
				n.push_back(std::make_pair((uint)i,(uint) j-1));
			}
			neighbors.insert(std::make_pair(p,n));
		}
	for(uint i = 0; i < (uint) image.rows; i++)
		for(uint j = 0; j < (uint) image.cols; j++)
		{
			std::pair<uint, uint> p(i,j);
			if(neighbors[p].size() == 0)
			{
				std::cerr << "no p should have 0 neighbors!\n";
				exit(1);
			} else if(neighbors[p].size() == 1)
			{
				std::cerr << "no p should have 1 neighbor!\n";
				exit(1);
			} else if(neighbors[p].size() > 4)
			{
				std::cerr << "no p should have > 4 neighbors!\n";
				exit(1);
			}
		}
	return neighbors;
}

HeightTable computeHeightTable(const cv::Mat& image)
{
	HeightTable ht;
	for(unsigned int i = 0; i < (uint) image.rows; i++)
		for(unsigned int j = 0; j < (uint) image.cols; j++)
		{
			if(image.at<uchar>(i,j) < 0 || image.at<uchar>(i,j) > 255)
			{
				std::cerr << "invalid: image.at<uchar>(i,j) in HeightTable!\n";
				exit(1);
			}
			ht[image.at<uchar>(i,j)].push_back(std::make_pair(i,j));
		}
	long sum = 0;
	for(uint i = 0; i < ht.size(); i++)
	{
		std::cout << i <<": " << ht[i].size() << "\n";
		sum += ht[i].size();
	}
	std::cout << "sum: " << sum << "| rowsXcols: " << image.rows * image.cols;
	std::cout << "\n\n";
	return ht;
}

cv::Mat watershedSegmentation(const cv::Mat& fi)
{
	/* initialization step */
	cv::Mat fo(fi.rows, fi.cols, CV_8UC1);

	std::vector<std::vector<int> > fo_new;
	fo_new.resize((uint) fi.rows);
	for(unsigned int i = 0; i < fo_new.size(); i++)
	{
		fo_new[i].resize((uint) fi.cols);
		for(uint j = 0; j < fo_new[i].size(); j++)
			fo_new[i][j] = init;
	}
	
	int current_label = 0;
	bool flag = true;
	Neighbors neighbors = computeNeighbors(fi);
	/***********************/

	std::cout << "finished with neighbor computing!\n";

	uchar h_min = fi.at<uchar>(0,0);
	uchar h_max = fi.at<uchar>(0,0);
	for(uint i = 0; i < (uint) fi.rows; i++)
		for(uint j = 0; j < (uint) fi.cols; j++)
		{
			if(fi.at<uchar>(i,j) < h_min) h_min = fi.at<uchar>(i,j);
			if(fi.at<uchar>(i,j) > h_max) h_max = fi.at<uchar>(i,j);
		}

	unsigned int hmin = (uint) h_min;
	unsigned int hmax = (uint) h_max;
	std::cout << "hmin: " << hmin << "\n";
	std::cout << "hmax: " << hmax << "\n";

	/* get height table for ease of access */
	HeightTable heights = computeHeightTable(fi);
	/***************************************/
	std::cout << "Finished with height table computation!\n";
	std::queue<std::pair<uint, uint> > fifo;
	/* main loop */
	for(unsigned int h = hmin; h <= hmax; h++)
	{
		/* 1st part */
		// mark with mask value all pixels at current height
		// and add to queue fifo if one of its neighbor
		// pixels was set before
		for(unsigned int i = 0; i < heights[h].size(); i++)
		{
			std::pair<uint, uint> p = heights[h][i];
			//fo.at<int>(p.first, p.second) = mask;
			fo_new[p.first][p.second] = mask;
			for(uint j = 0; j < neighbors[p].size(); j++)
			{
				std::pair<uint, uint> p1 = neighbors[p][j];
				if(fo_new[p.first][p.second] > 0 || 
					fo_new[p.first][p.second] == wshed)
				{
					fo_new[p.first][p.second] = inqueue;
					fifo.push(p1);
				}
			}
		}

		//std::cout << "First part for h: " << h << "\n";
		/* 2nd part */
		while(!fifo.empty())
		{
			std::pair<uint, uint> p = fifo.front();
			fifo.pop();
			// for all the neighbors of pixel p
			for(uint i = 0; i < neighbors[p].size(); i++)
			{
				std::pair<uint, uint> p1 = neighbors[p][i];
				if(fo_new[p.first][p.second] > 0)
				{
					// i.e. p1 belongs to an already labelled basin
					// case 1: p is not set or p is watershed and flag is true
					//         p is set to lable of p1
					if(fo_new[p.first][p.second] == inqueue ||
						(fo_new[p.first][p.second] == wshed &&
							flag == true))
					{
						fo_new[p.first][p.second] = 
							fo_new[p1.first][p1.second];
					}
					// case 2: p has already a label but its different than p1
					// 		   in such case set p to wshed and flag to false
					else if(fo_new[p.first][p.second] > 0 &&
						(fo_new[p.first][p.second] != 
							fo_new[p1.first][p1.second]))
					{
						fo_new[p.first][p.second] = wshed;
						flag = false;
					}
				} else if(fo_new[p1.first][p1.second] == wshed)
				{
					// if neighbor is set to wshed value 
					// check if current is not set, if yes set it to
					// wshed as well
					if(fo_new[p.first][p.second] == inqueue)
					{
						fo_new[p.first][p.second] = wshed;
						flag = true;
					}
				} else if(fo_new[p1.first][p1.second] == mask)
				{
					// if neighbor equals mask add neighbor to queue
					fo_new[p.first][p.second] = inqueue;
					fifo.push(p1);
				}
			}
		}
		//std::cout << "Second part for h: " << h << "\n";

		/* 3rd part */
		for(unsigned int i = 0; i < heights[h].size(); i++)
		{
			// check for new minima
			std::pair<uint, uint> p = heights[h][i];
			if(fo_new[p.first][p.second] == mask)
			{
				current_label++;
				fifo.push(p);
				fo_new[p.first][p.second] = current_label;
				// propagate the current_label trough the 
				// connected structure given by neighbors a.k.a BFS
				while(!fifo.empty())
				{
					std::pair<uint, uint> p1 = fifo.front();
					fifo.pop();
					if(neighbors.find(p1) != neighbors.end())
					{
						for(uint j = 0; j < neighbors[p1].size(); j++)
						{
							std::pair<uint, uint> p2 = neighbors[p1][j];
							if(p2.first < 0 || p2.first >= (uint) fi.rows || 
									p2.second < 0 || p2.second >= (uint) fi.cols)
							{
								std::cout << "p2.first: " << p2.first << "\n";
								std::cout << "p2.second: " << p2.second << "\n";
								std::cout << "Should not happen!\n";
							}
							if(fo_new[p2.first][p2.second] == mask)
							{
								fifo.push(p2);
								fo_new[p2.first][p2.second] = current_label;
							}
						}
					} else {
						std::cerr << "pair (" << p1.first << ", " << p1.second;
						std::cerr << ")" << " not found in neighbors map!\n";
						exit(1); 
					}
				}
			}
		}
	}

	int max_fo_new = fo_new[0][0];
	for(uint i = 0; i < (uint)fo.rows; i++)
		for(uint j = 0; j < (uint)fo.cols; j++)
		{
			if(max_fo_new < fo_new[0][0]) max_fo_new = fo_new[0][0];
		}
	std::cout << "max_fo_new: " << max_fo_new << "\n";
	std::cout << "current_label: " << current_label << "\n";

	for(uint i = 0; i < (uint)fo.rows; i++)
		for(uint j = 0; j < (uint)fo.cols; j++)
		{
			if(fo_new[i][j] < 0)
			{
				std::cerr << "fo_new values should be >= than 0(wshed) !\n";
				exit(1);
			}
			fo.at<uchar>(i,j) = (uchar) fo_new[i][j];
		}

	return fo;
}

int main(int argc, char **argv)
{
	if(argc <= 1)
	{
		showHelp(argv[0]);
		exit(1);
	}

	char tmp;
	std::string inputFilePath("../input/");
	std::string inputImageName;
	std::string outputFilePath("../output/");
	std::string outputImageName;
	bool showImage = false;
	/* using getopt to parse command line arguments */
	/* h option without argument, i,o options with argument*/
	while((tmp = getopt(argc, argv, "hi:o:s")) != -1)
	{
		switch(tmp)
		{
			case 'h':
				showHelp(argv[0]);
				exit(0);
			case 'i':
				inputImageName = optarg;
				break;
			case 'o':
				outputImageName = optarg;
				break;
			case 's':
				showImage = true;
				break;
			case '?':
				if(optopt == 'i' || optopt == 'o')
				{
					std::cerr << "Option -" << optopt << " requires and argument.\n";
				} else if(isprint(optopt))
				{
					std::cerr << "Unknown option `-" << optopt << "'.\n";
				} else
				{
					std::cerr << "Unknown option character `\\x" << std::hex << optopt;
					std::cerr << "\n";
				}
				exit(1);
			default: 
				showHelp(argv[0]);
				break;
		}
	}

	/* check if filepath(s) can be opened */
	if(inputImageName.empty())
	{
		showHelp(argv[0]);
		exit(1);
	}

	/* input image */
	cv::Mat image;

	if(!inputImageName.empty())
	{
		std::ifstream inputImageFile((inputFilePath + inputImageName).c_str());
		if(!inputImageFile.good())
		{
			std::cerr << "There was a problem opening: ";
			std::cerr << inputFilePath + inputImageName << "\n";
			exit(1);
		}
		inputImageFile.close();
		image = readImage(inputFilePath + inputImageName);
		std::cout << "rows: " << image.rows << "\n";
		std::cout << "cols: " << image.cols << "\n";
	}
	/**************************/

	/* segmentation call */
	cv::Mat segmented_image = watershedSegmentation(image);
	if(!segmented_image.data)
	{
		std::cerr << "The segmented image is empty!\n";
		exit(1);
	} else {
		std::cout << "Segmented image was computed!\n";
	}
	/*********************/

	/* if user wants to output the image */
	if(!outputImageName.empty())
	{
		std::ofstream outputImageFile((outputFilePath + outputImageName).c_str());
		if(!outputImageFile.good())
		{
			std::cerr << "There was a problem opening: ";
			std::cerr << outputFilePath + outputImageName << "\n";
			exit(1);
		}
		outputImageFile.close();
		writeImage(outputFilePath + outputImageName, segmented_image);
	} else {
		outputImageName = "out_"+currentDateTime()+".jpg";
		writeImage(outputFilePath + outputImageName, segmented_image);
	}
	/**************************/

	/* visualize image */
	if(showImage)
	{
		// Create a window for display.
		cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );
		// Show our image inside it.
		cv::imshow( "Display window", segmented_image );
		// Wait for a keystroke in the window
		cv::waitKey(0);                                          
	}
	/*******************/

	return 0;
}
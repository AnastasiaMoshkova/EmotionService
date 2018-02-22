///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// THIS SOFTWARE IS PROVIDED УAS ISФ FOR ACADEMIC USE ONLY AND ANY EXPRESS
// OR IMPLIED WARRANTIES WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY.
// OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Notwithstanding the license granted herein, Licensee acknowledges that certain components
// of the Software may be covered by so-called Уopen sourceФ software licenses (УOpen Source
// ComponentsФ), which means any software licenses approved as open source licenses by the
// Open Source Initiative or any substantially similar licenses, including without limitation any
// license that, as a condition of distribution of the software licensed under such license,
// requires that the distributor make the software available in source code format. Licensor shall
// provide a list of Open Source Components for a particular version of the Software upon
// LicenseeТs request. Licensee will comply with the applicable terms of such licenses and to
// the extent required by the licenses covering Open Source Components, the terms of such
// licenses will apply in lieu of the terms of this Agreement. To the extent the terms of the
// licenses applicable to Open Source Components prohibit any of the restrictions in this
// License Agreement with respect to such Open Source Component, such restrictions will not
// apply to such Open Source Component. To the extent the terms of the licenses applicable to
// Open Source Components require Licensor to make an offer to provide source code or
// related information in connection with the Software, such offer is hereby made. Any request
// for source code or related information should be directed to cl-face-tracker-distribution@lists.cam.ac.uk
// Licensee acknowledges receipt of notices for the Open Source Components for the initial
// delivery of the Software.

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace: an open source facial behavior analysis toolkit
//       Tadas BaltruЪaitis, Peter Robinson, and Louis-Philippe Morency
//       in IEEE Winter Conference on Applications of Computer Vision, 2016  
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas BaltruЪaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-speci?c normalisation for automatic Action Unit detection
//       Tadas BaltruЪaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       Tadas BaltruЪaitis, Peter Robinson, and Louis-Philippe Morency. 
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////


// FeatureExtraction.cpp : Defines the entry point for the feature extraction console application.
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <stdio.h>
#include <winsock2.h> // Wincosk2.h должен быть раньше windows!
#include <cstdlib>
#include <iostream>
#include <winsock.h>

#include <vector>
// System includes
#include <fstream>
#include <sstream>

#include<vector>
#include <iostream>
#include <algorithm>    
#include <Windows.h>
#include <iterator>
#include <numeric>

// OpenCV includes
#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>
#include <boost/algorithm/string.hpp>

// Local includes
#include "LandmarkCoreIncludes.h"

#include <Face_utils.h>
#include <FaceAnalyser.h>
#include <GazeEstimation.h>


// Need to link with Ws2_32.lib
#pragma comment (lib, "Ws2_32.lib")
// #pragma comment (lib, "Mswsock.lib")

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort(const std::string & error)
{
	std::cout << error << std::endl;
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )



using namespace std;
using namespace cv;

using namespace boost::filesystem;

vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	// First argument is reserved for the name of the executable
	for (int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

// Useful utility for creating directories for storing the output files
void create_directory_from_file(string output_path)
{

	// Creating the right directory structure

	// First get rid of the file
	auto p = path(path(output_path).parent_path());

	if (!p.empty() && !boost::filesystem::exists(p))
	{
		bool success = boost::filesystem::create_directories(p);
		if (!success)
		{
			cout << "Failed to create a directory... " << p.string() << endl;
		}
	}
}

void create_directory(string output_path)
{

	// Creating the right directory structure
	auto p = path(output_path);

	if (!boost::filesystem::exists(p))
	{
		bool success = boost::filesystem::create_directories(p);

		if (!success)
		{
			cout << "Failed to create a directory..." << p.string() << endl;
		}
	}
}

void get_output_feature_params(vector<string> &output_similarity_aligned, vector<string> &output_hog_aligned_files, double &similarity_scale,
	int &similarity_size, bool &grayscale, bool& verbose, bool& dynamic, bool &output_2D_landmarks, bool &output_3D_landmarks,
	bool &output_model_params, bool &output_pose, bool &output_AUs, bool &output_gaze, vector<string> &arguments);

void get_image_input_output_params_feats(vector<vector<string> > &input_image_files, bool& as_video, vector<string> &arguments);

void output_HOG_frame(std::ofstream* hog_file, bool good_frame, const cv::Mat_<double>& hog_descriptor, int num_rows, int num_cols);

// Some globals for tracking timing information for visualisation
double fps_tracker = -1.0;
int64 t0 = 0;

// Visualising the results
void visualise_tracking(cv::Mat& captured_image, const LandmarkDetector::CLNF& face_model, const LandmarkDetector::FaceModelParameters& det_parameters, cv::Point3f gazeDirection0, cv::Point3f gazeDirection1, int frame_count, double fx, double fy, double cx, double cy)
{

	// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
	double detection_certainty = face_model.detection_certainty;
	bool detection_success = face_model.detection_success;

	double visualisation_boundary = 0.2;

	// Only draw if the reliability is reasonable, the value is slightly ad-hoc
	if (detection_certainty < visualisation_boundary)
	{
		LandmarkDetector::Draw(captured_image, face_model);

		double vis_certainty = detection_certainty;
		if (vis_certainty > 1)
			vis_certainty = 1;
		if (vis_certainty < -1)
			vis_certainty = -1;

		vis_certainty = (vis_certainty + 1) / (visualisation_boundary + 1);

		// A rough heuristic for box around the face width
		int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

		cv::Vec6d pose_estimate_to_draw = LandmarkDetector::GetCorrectedPoseWorld(face_model, fx, fy, cx, cy);

		// Draw it in reddish if uncertain, blueish if certain
		LandmarkDetector::DrawBox(captured_image, pose_estimate_to_draw, cv::Scalar((1 - vis_certainty)*255.0, 0, vis_certainty * 255), thickness, fx, fy, cx, cy);

		if (det_parameters.track_gaze && detection_success && face_model.eye_model)
		{
			FaceAnalysis::DrawGaze(captured_image, face_model, gazeDirection0, gazeDirection1, fx, fy, cx, cy);
		}
	}

	// Work out the framerate
	if (frame_count % 10 == 0)
	{
		double t1 = cv::getTickCount();
		fps_tracker = 10.0 / (double(t1 - t0) / cv::getTickFrequency());
		t0 = t1;
	}

	// Write out the framerate on the image before displaying it
	char fpsC[255];
	std::sprintf(fpsC, "%d", (int)fps_tracker);
	string fpsSt("FPS:");
	fpsSt += fpsC;
	cv::putText(captured_image, fpsSt, cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 1, CV_AA);

	if (!det_parameters.quiet_mode)
	{
		//cv::namedWindow("tracking_result", 1);
		//cv::imshow("tracking_result", captured_image);
	}
}

void prepareOutputFile(std::ofstream* output_file, bool output_2D_landmarks, bool output_3D_landmarks,
	bool output_model_params, bool output_pose, bool output_AUs, bool output_gaze,
	int num_landmarks, int num_model_modes, vector<string> au_names_class, vector<string> au_names_reg);

// Output all of the information into one file in one go (quite a few parameters, but simplifies the flow)
void outputAllFeatures(std::ofstream* output_file, bool output_2D_landmarks, bool output_3D_landmarks,
	bool output_model_params, bool output_pose, bool output_AUs, bool output_gaze,
	const LandmarkDetector::CLNF& face_model, int frame_count, double time_stamp, bool detection_success,
	cv::Point3f gazeDirection0, cv::Point3f gazeDirection1, const cv::Vec6d& pose_estimate, double fx, double fy, double cx, double cy,
	const FaceAnalysis::FaceAnalyser& face_analyser, cv::Mat& captured_image, std::vector<string> &vec);

void outputAllF(const FaceAnalysis::FaceAnalyser& face_analyser, cv::Mat& captured_image, std::vector<string> &vec);

void post_process_output_file(FaceAnalysis::FaceAnalyser& face_analyser, string output_file, bool dynamic);


int main(int argc, char **argv)
{

	vector<string> arguments = get_arguments(argc, argv);

	// Some initial parameters that can be overriden from command line	
	vector<string> input_files, depth_directories, output_files, tracked_videos_output;

	LandmarkDetector::FaceModelParameters det_parameters(arguments);
	// Always track gaze in feature extraction
	det_parameters.track_gaze = true;

	// Get the input output file parameters

	// Indicates that rotation should be with respect to camera or world coordinates
	bool use_world_coordinates;
	LandmarkDetector::get_video_input_output_params(input_files, depth_directories, output_files, tracked_videos_output, use_world_coordinates, arguments);

	bool video_input = true;
	bool verbose = true;
	bool images_as_video = false;

	vector<vector<string> > input_image_files;

	// Adding image support for reading in the files
	if (input_files.empty())
	{
		vector<string> d_files;
		vector<string> o_img;
		vector<cv::Rect_<double>> bboxes;
		get_image_input_output_params_feats(input_image_files, images_as_video, arguments);

		if (!input_image_files.empty())
		{
			video_input = false;
		}

	}

	// Grab camera parameters, if they are not defined (approximate values will be used)
	float fx = 0, fy = 0, cx = 0, cy = 0;
	int d = 0;
	// Get camera parameters
	LandmarkDetector::get_camera_params(d, fx, fy, cx, cy, arguments);

	// If cx (optical axis centre) is undefined will use the image size/2 as an estimate
	bool cx_undefined = false;
	bool fx_undefined = false;
	if (cx == 0 || cy == 0)
	{
		cx_undefined = true;
	}
	if (fx == 0 || fy == 0)
	{
		fx_undefined = true;
	}

	// The modules that are being used for tracking
	LandmarkDetector::CLNF face_model(det_parameters.model_location);

	vector<string> output_similarity_align;
	vector<string> output_hog_align_files;

	double sim_scale = 0.7;
	int sim_size = 112;
	bool grayscale = false;
	bool video_output = false;
	bool dynamic = true; // Indicates if a dynamic AU model should be used (dynamic is useful if the video is long enough to include neutral expressions)
	int num_hog_rows;
	int num_hog_cols;

	// By default output all parameters, but these can be turned off to get smaller files or slightly faster processing times
	// use -no2Dfp, -no3Dfp, -noMparams, -noPose, -noAUs, -noGaze to turn them off
	bool output_2D_landmarks = false;
	bool output_3D_landmarks = false;
	bool output_model_params = false;
	bool output_pose = false;
	bool output_AUs = true;
	bool output_gaze = false;

	get_output_feature_params(output_similarity_align, output_hog_align_files, sim_scale, sim_size, grayscale, verbose, dynamic,
		output_2D_landmarks, output_3D_landmarks, output_model_params, output_pose, output_AUs, output_gaze, arguments);

	// Used for image masking

	string tri_loc;
	if (boost::filesystem::exists(path("model/tris_68_full.txt")))
	{
		tri_loc = "model/tris_68_full.txt";
	}
	else
	{
		path loc = path(arguments[0]).parent_path() / "model/tris_68_full.txt";
		tri_loc = loc.string();

		if (!exists(loc))
		{
			cout << "Can't find triangulation files, exiting" << endl;
			return 1;
		}
	}

	// Will warp to scaled mean shape
	cv::Mat_<double> similarity_normalised_shape = face_model.pdm.mean_shape * sim_scale;
	// Discard the z component
	similarity_normalised_shape = similarity_normalised_shape(cv::Rect(0, 0, 1, 2 * similarity_normalised_shape.rows / 3)).clone();

	// If multiple video files are tracked, use this to indicate if we are done
	bool done = false;
	int f_n = -1;
	int curr_img = -1;

	string au_loc;

	string au_loc_local;
	if (dynamic)
	{
		au_loc_local = "AU_predictors/AU_all_best.txt";
	}
	else
	{
		au_loc_local = "AU_predictors/AU_all_static.txt";
	}

	if (boost::filesystem::exists(path(au_loc_local)))
	{
		au_loc = au_loc_local;
	}
	else
	{
		path loc = path(arguments[0]).parent_path() / au_loc_local;

		if (exists(loc))
		{
			au_loc = loc.string();
		}
		else
		{
			cout << "Can't find AU prediction files, exiting" << endl;
			return 1;
		}
	}

	// Creating a  face analyser that will be used for AU extraction
	FaceAnalysis::FaceAnalyser face_analyser(vector<cv::Vec3d>(), 0.7, 112, 112, au_loc, tri_loc);

	///////////////
	
	char client_name[30];
	char sv_name[30];

	WSADATA WsaData;
	if (int err = WSAStartup(MAKEWORD(2, 0), &WsaData) != 0) {
		std::cout << "Socket not Loaded!n";
	}
	else {
		std::cout << "Socket Loaded  n";
	}


	gethostname(sv_name, 30);

	int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP); 
	if (sock == -1) { 
		std::cout << "Error! Socket no created.\n";
	}
	else {
		std::cout << "Socket Create.\n";
	}

	sockaddr_in addr;
	addr.sin_family = AF_INET;
	addr.sin_port = htons(8080); 
	//addr.sin_addr.s_addr = htonl(INADDR_ANY); 
	addr.sin_addr.s_addr = inet_addr("127.0.0.1");

	int bindet = ::bind(sock, (sockaddr *)&addr, sizeof(addr));
	if (bindet == -1) {
		std::cout << "Binding Error!n";

		std::cout << "Number1\n";

		//system("pause");
	}
	//else {
		std::cout << "Number1\n";
		int listening = listen(sock, 100);

		std::cout << "Server Name: " << sv_name << endl << "Wait for connecting ...\n";
		int acc = accept(sock, (sockaddr*)&addr, 0);
	
		
		int bytes_recv;
	
		
		// ќбъ€вл€ем переменные
		int iResult;                                                                                // ѕеременна€ на результат операций
		const int MAX_BUF_SIZE = 2073600;                                     // ѕроизвольно максимальный размер приемного буфера
		unsigned char *buf = new unsigned char[MAX_BUF_SIZE];   // Ѕуфер дл€ прима сообщений
		vector<uchar> videoBuffer;                                                     // Ѕуфер данных изображени€
		Mat jpegimage;                                                                       // ¬ектор данных изображени€
		IplImage img;

			string current_file;

			cv::VideoCapture video_capture;

			cv::Mat captured_image;
			int total_frames = -1;
			int reported_completion = 0;

			double fps_vid_in = -1.0;

	
				// Creating output files
				std::ofstream output_file;

	
				// Saving the HOG features
				std::ofstream hog_output_file;
				
				int frame_count = 0;

				// This is useful for a second pass run (if want AU predictions)
				vector<cv::Vec6d> params_global_video;
				vector<bool> successes_video;
				vector<cv::Mat_<double>> params_local_video;
				vector<cv::Mat_<double>> detected_landmarks_video;

				// Use for timestamping if using a webcam
				int64 t_initial = cv::getTickCount();

				bool visualise_hog = verbose;

				// Timestamp in seconds of current processing
				double time_stamp = 0;
			
				vector<string> vv;
				int ii = 0;
				string st1, st2, st3, st4;


				double x1, x2, x3, x4, x5, x6, x7;
				int key1 = 0, key2 = 0, key3 = 0, key4 = 0, key5 = 0, key6 = 0, key7 = 0;
				INFO_STREAM("Starting tracking");
					
					f_n = 0;
				//vector<string> vv;
				if (!output_files.empty())
				{
					std::cout << "Number7\n";
					output_file.open(output_files[f_n], ios_base::out);
					prepareOutputFile(&output_file, output_2D_landmarks, output_3D_landmarks, output_model_params, output_pose, output_AUs, output_gaze, face_model.pdm.NumberOfPoints(), face_model.pdm.NumberOfModes(), face_analyser.GetAUClassNames(), face_analyser.GetAURegNames());
				}

		
				
				while (((iResult = recv(acc, (char *)&buf[0], MAX_BUF_SIZE, 0)) > 0) && (done = true))
					
					{
						std::cout << "Number2\n";
						// ≈сли пришли данные изображени€, копируем их
						videoBuffer.resize(iResult);
						memcpy((char*)(&videoBuffer[0]), buf, iResult);
						// ƒекодируем данные
						jpegimage = imdecode(Mat(videoBuffer), CV_LOAD_IMAGE_COLOR);
						img = jpegimage;


						// ¬ыводим изображение
					
						captured_image = jpegimage;

			
						char c = cvWaitKey(10); //∆дем нажати€ кнопки и записываем нажатую кнопку в переменную с.
						if (c == 113 || c == 81) //ѕровер€ем, кака€ кнопка нажата. 113 и 81 - это коды кнопки "q" - в английской и русской раскладках. 
						{
							done = false;
						
							return 0;  //выходит из программы. 
						}
					
					std::cout << "Number10\n";

					std::cout << "Number5\n";
					cx = captured_image.cols / 2.0f;
					cy = captured_image.rows / 2.0f;

					std::cout << "Number6\n";
					fx = 500 * (captured_image.cols / 640.0);
					fy = 500 * (captured_image.rows / 480.0);

					fx = (fx + fy) / 2.0;
					fy = fx;

					std::cout << "Number100\n";
					// Grab the timestamp first
			
					time_stamp = (double)frame_count * (1.0 / 30.0);
					// Reading the images
					cv::Mat_<uchar> grayscale_image;

					if (captured_image.channels() == 3)
					{
						cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);
					}
					else
					{
						grayscale_image = captured_image.clone();
					}
					std::cout << "Number101\n";
					// The actual facial landmark detection / tracking
					bool detection_success;

					
					detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, face_model, det_parameters);
					// Gaze tracking, absolute gaze direction

					std::cout << "Number101_1\n";
					cv::Point3f gazeDirection0(0, 0, -1);
					cv::Point3f gazeDirection1(0, 0, -1);

					if (det_parameters.track_gaze && detection_success && face_model.eye_model)
					{
						FaceAnalysis::EstimateGaze(face_model, gazeDirection0, fx, fy, cx, cy, true);
						FaceAnalysis::EstimateGaze(face_model, gazeDirection1, fx, fy, cx, cy, false);
					}

					std::cout << "Number102\n";
					// Do face alignment
					cv::Mat sim_warped_img;
					cv::Mat_<double> hog_descriptor;

					// But only if needed in output
					if (!output_similarity_align.empty() || hog_output_file.is_open() || output_AUs)
					{
						face_analyser.AddNextFrame(captured_image, face_model, time_stamp, false, !det_parameters.quiet_mode);
						face_analyser.GetLatestAlignedFace(sim_warped_img);
						//outputAllF(face_analyser, captured_image, vv);
						if (!det_parameters.quiet_mode)
						{
							
						}
						if (hog_output_file.is_open())
						{
							FaceAnalysis::Extract_FHOG_descriptor(hog_descriptor, sim_warped_img, num_hog_rows, num_hog_cols);

							if (visualise_hog && !det_parameters.quiet_mode)
							{
								cv::Mat_<double> hog_descriptor_vis;
								FaceAnalysis::Visualise_FHOG(hog_descriptor, num_hog_rows, num_hog_cols, hog_descriptor_vis);
								
							}
						}
					}
					std::cout << "Number103\n";
					// Work out the pose of the head from the tracked model
					cv::Vec6d pose_estimate;
					if (use_world_coordinates)
					{
						pose_estimate = LandmarkDetector::GetCorrectedPoseWorld(face_model, fx, fy, cx, cy);
					}
					else
					{
						pose_estimate = LandmarkDetector::GetCorrectedPoseCamera(face_model, fx, fy, cx, cy);
					}

					if (hog_output_file.is_open())
					{
						output_HOG_frame(&hog_output_file, detection_success, hog_descriptor, num_hog_rows, num_hog_cols);
					}

					// Write the similarity normalised output
					if (!output_similarity_align.empty())
					{
						std::cout << "Number_100" << std::endl;
						if (sim_warped_img.channels() == 3 && grayscale)
						{
							cvtColor(sim_warped_img, sim_warped_img, CV_BGR2GRAY);
						}

						char name[100];

						// output the frame number
						std::sprintf(name, "frame_det_%06d.bmp", frame_count);

						// Construct the output filename
						boost::filesystem::path slash("/");

						std::string preferredSlash = slash.make_preferred().string();

						string out_file = output_similarity_align[f_n] + preferredSlash + string(name);
						bool write_success = imwrite(out_file, sim_warped_img);

						if (!write_success)
						{
							cout << "Could not output similarity aligned image image" << endl;
							return 1;
						}
					}

					
					// Output the landmarks, pose, gaze, parameters and AUs
					outputAllFeatures(&output_file, output_2D_landmarks, output_3D_landmarks, output_model_params, output_pose, output_AUs, output_gaze,
						face_model, frame_count, time_stamp, detection_success, gazeDirection0, gazeDirection1,
						pose_estimate, fx, fy, cx, cy, face_analyser, captured_image, vv);


					

					// Work out the framerate
					if (frame_count % 5 == 0) {
						
						st1 = vv[vv.size() - 1];
						st2 = vv[vv.size() - 2];
						st3 = vv[vv.size() - 3];
						st4 = vv[vv.size() - 4];
					}


					//////////
					if ((vv[0] == "surprise") || (vv[1] == "surprise"))
					{
						key1++;
					}
					if ((vv[0] == "fear") || (vv[1] == "fear"))
					{
						key2++;
					}
					if ((vv[0] == "happiness") || (vv[1] == "happiness"))
					{
						key3++;
					}
					if ((vv[0] == "sad") || (vv[1] == "sad"))
					{
						key4++;
					}
					if ((vv[0] == "disgust") || (vv[1] == "disgust"))
					{
						key5++;
					}
					if ((vv[0] == "fear") || (vv[1] == "fear"))
					{
						key6++;
					}
					if ((vv[0] == "neutral") || (vv[1] == "neutral"))
					{
						key7++;
					}
					/////////
					cv::putText(captured_image,
						//f[f.size()-1],
						st1,
						cv::Point(200, 100), // Coordinates
						cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
						1.0, // Scale. 2.0 = 2x bigger
						cv::Scalar(255, 0, 0), // Color
						1, // Thickness
						CV_AA); // Anti-alias
					cv::putText(captured_image,
						//f[f.size()-1],
						st2,
						cv::Point(200, 50), // Coordinates
						cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
						1.0, // Scale. 2.0 = 2x bigger
						cv::Scalar(255, 0, 0), // Color
						1, // Thickness
						CV_AA); // Anti-alias
					cv::putText(captured_image,
						//f[f.size()-1],
						st3,
						cv::Point(50, 100), // Coordinates
						cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
						1.0, // Scale. 2.0 = 2x bigger
						cv::Scalar(255, 0, 0), // Color
						1, // Thickness
						CV_AA); // Anti-alias
					cv::putText(captured_image,
						//f[f.size()-1],
						st4,
						cv::Point(50, 50), // Coordinates
						cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
						1.0, // Scale. 2.0 = 2x bigger
						cv::Scalar(255, 0, 0), // Color
						1, // Thickness
						CV_AA); // Anti-alias

					cv::namedWindow("server_camera", 1);
					cv::imshow("server_camera", captured_image);


					// output the tracked video
					if (!tracked_videos_output.empty())
					{
						//writerFace << captured_image;
					}

					if (video_input)
					{
						video_capture >> captured_image;
					}
					else
					{
						curr_img++;
						if (curr_img < (int)input_image_files[f_n].size())
						{
							string curr_img_file = input_image_files[f_n][curr_img];
							captured_image = cv::imread(curr_img_file, -1);
						}
						else
						{
							captured_image = cv::Mat();
						}
					}


					// detect key presses
					char character_press = cv::waitKey(1);

					// restart the tracker
					if (character_press == 'r')
					{
						face_model.Reset();
					}
					// quit the application
					else if (character_press == 'q')
					{
						return(0);
					}

					// Update the frame count
					frame_count++;
					std::cout << "FRAME - "<<frame_count << std::endl;


					////////////////
					Mat image = Mat::zeros(900, 1100, CV_8UC3);
					//int x1 = 0, x2 = 0, x3 = 0, x4 = 0, x5 = 0, x6 = 0, x7 = 0;
					x1 = (key7 * 100) / (frame_count * 2);
					x2 = (key1 * 100) / (frame_count * 2);
					x3 = (key2 * 100) / (frame_count * 2);
					x4 = (key3 * 100) / (frame_count * 2);
					x5 = (key4 * 100) / (frame_count * 2);
					x6 = (key5 * 100) / (frame_count * 2);
					x7 = (key6 * 100) / (frame_count * 2);
					x1 = round(x1 * 600 / 100);
					x1 = 750 - x1;
					x2 = round(x2 * 600 / 100);
					x2 = 750 - x2;
					x3 = round(x3 * 600 / 100);
					x3 = 750 - x3;
					x4 = round(x4 * 600 / 100);
					x4 = 750 - x4;
					x5 = round(x5 * 600 / 100);
					x5 = 750 - x5;
					x6 = round(x6 * 600 / 100);
					x6 = 750 - x6;
					x7 = round(x7 * 600 / 100);
					x7 = 750 - x7;

					std::cout << "x1" << x1 << "  " << x2 << "  " << x3 << std::endl;
					//x = 100;
					// Draw a circle 
					//circle(image, Point(200, 200), 32.0, Scalar(0, 0, 255), 1, 8);
					line(image, Point(75, 750), Point(75, x1), Scalar(110, 220, 0), 40, 8);
					line(image, Point(225, 750), Point(225, x2), Scalar(50, 70, 0), 40, 8);
					line(image, Point(375, 750), Point(375, x3), Scalar(150, 40, 0), 40, 8);
					line(image, Point(375 + 150, 750), Point(375 + 150, x4), Scalar(110, 220, 0), 40, 8);
					line(image, Point(375 + 300, 750), Point(375 + 300, x5), Scalar(50, 70, 0), 40, 8);
					line(image, Point(375 + 450, 750), Point(375 + 450, x6), Scalar(150, 40, 0), 40, 8);
					line(image, Point(375 + 600, 750), Point(375 + 600, x7), Scalar(150, 40, 0), 40, 8);
					putText(image, "neutral", Point(0, 800), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 200, 200), 1);
					putText(image, "surprise", Point(150, 800), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 200, 200), 1);
					putText(image, "fear", Point(300, 800), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 200, 200), 1);
					putText(image, "happiness", Point(450, 800), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 200, 200), 1);
					putText(image, "sadness", Point(600, 800), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 200, 200), 1);
					putText(image, "disgust", Point(750, 800), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 200, 200), 1);
					putText(image, "anger", Point(900, 800), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 200, 200), 1);

					
					imshow("Emotion", image);
					////////////////
					if (total_frames != -1)
					{
						if ((double)frame_count / (double)total_frames >= reported_completion / 10.0)
						{
							cout << reported_completion * 10 << "% ";
							reported_completion = reported_completion + 1;
						}
					}

				}

				output_file.close();

				if (output_files.size() > 0 && output_AUs)
				{
					cout << "Postprocessing the Action Unit predictions" << endl;
					//post_process_output_file(face_analyser, output_files[f_n], dynamic);
					cv::putText(captured_image,
						"Here is some text",
						cv::Point(5, 5), // Coordinates
						cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
						1.0, // Scale. 2.0 = 2x bigger
						cv::Scalar(255, 255, 255), // Color
						1, // Thickness
						CV_AA); // Anti-alias
				}
				// Reset the models for the next video
				face_analyser.Reset();
				face_model.Reset();

				frame_count = 0;
				curr_img = -1;

				if (total_frames != -1)
				{
					cout << endl;
				}

				// break out of the loop if done with all the files (or using a webcam)
				if ((video_input && f_n == input_files.size() - 1) || (!video_input && f_n == input_image_files.size() - 1))
				{
					//done = true;
				}
			//	}
			//}
			//}
		//}

	return 0;
}

// Allows for post processing of the AU signal
void post_process_output_file(FaceAnalysis::FaceAnalyser& face_analyser, string output_file, bool dynamic)
{

	vector<double> certainties;
	vector<bool> successes;
	vector<double> timestamps;
	vector<std::pair<std::string, vector<double>>> predictions_reg;
	vector<std::pair<std::string, vector<double>>> predictions_class;

	// Construct the new values to overwrite the output file with
	face_analyser.ExtractAllPredictionsOfflineReg(predictions_reg, certainties, successes, timestamps, dynamic);
	face_analyser.ExtractAllPredictionsOfflineClass(predictions_class, certainties, successes, timestamps, dynamic);

	int num_class = predictions_class.size();
	int num_reg = predictions_reg.size();

	// Extract the indices of writing out first
	vector<string> au_reg_names = face_analyser.GetAURegNames();
	std::sort(au_reg_names.begin(), au_reg_names.end());
	vector<int> inds_reg;

	// write out ar the correct index
	for (string au_name : au_reg_names)
	{
		for (int i = 0; i < num_reg; ++i)
		{
			if (au_name.compare(predictions_reg[i].first) == 0)
			{
				inds_reg.push_back(i);
				break;
			}
		}
	}

	vector<string> au_class_names = face_analyser.GetAUClassNames();
	std::sort(au_class_names.begin(), au_class_names.end());
	vector<int> inds_class;

	// write out ar the correct index
	for (string au_name : au_class_names)
	{
		for (int i = 0; i < num_class; ++i)
		{
			if (au_name.compare(predictions_class[i].first) == 0)
			{
				inds_class.push_back(i);
				break;
			}
		}
	}
	// Read all of the output file in
	vector<string> output_file_contents;

	std::ifstream infile(output_file);
	string line;

	while (std::getline(infile, line))
		output_file_contents.push_back(line);

	infile.close();

	// Read the header and find all _r and _c parts in a file and use their indices
	std::vector<std::string> tokens;
	boost::split(tokens, output_file_contents[0], boost::is_any_of(","));

	int begin_ind = -1;

	for (size_t i = 0; i < tokens.size(); ++i)
	{
		if (tokens[i].find("AU") != string::npos && begin_ind == -1)
		{
			begin_ind = i;
			break;
		}
	}
	int end_ind = begin_ind + num_class + num_reg;

	// Now overwrite the whole file
	std::ofstream outfile(output_file, ios_base::out);
	// Write the header
	outfile << output_file_contents[0].c_str() << endl;

	// Write the contents
	vector<double> aa;
	for (int i = 1; i < (int)output_file_contents.size(); ++i)
	{
		std::vector<std::string> tokens;
		boost::split(tokens, output_file_contents[i], boost::is_any_of(","));

		outfile << tokens[0];

		for (int t = 1; t < (int)tokens.size(); ++t)
		{
			if (t >= begin_ind && t < end_ind)
			{
				if (t - begin_ind < num_reg)
				{
					//outfile << ", " << predictions_reg[inds_reg[t - begin_ind]].second[i - 1];
					aa.push_back(predictions_reg[inds_reg[t - begin_ind]].second[i - 1]);
				}
				else
				{
					//	outfile << ", " << predictions_class[inds_class[t - begin_ind - num_reg]].second[i - 1];
					aa.push_back(predictions_class[inds_class[t - begin_ind - num_reg]].second[i - 1]);
				}
			}
			else
			{
				//	outfile << ", " << tokens[t];
			}

		}
		for (int t1 = 0; t1 < aa.size(); t1++)
		{
			//outfile << aa[t1];
		}

		vector<double> e1, e2, e3, e4, e5, e6;
		
		double AU01_c = aa[0], AU02_c = aa[1], AU04_c = aa[2], AU05_c = aa[3], AU06_c = aa[4], AU07_c = aa[5], AU09_c = aa[6],
			AU10_c = aa[7], AU12_c = aa[8], AU14_c = aa[9], AU15_c = aa[10], AU17_c = aa[11], AU20_c = aa[12], AU23_c = aa[13],
			AU25_c = aa[14], AU26_c = aa[15], AU45_c = aa[16];
		double AU01_r = aa[17], AU02_r = aa[18], AU04_r = aa[19], AU05_r = aa[20], AU06_r = aa[21], AU07_r = aa[22], AU09_r = aa[23],
			AU10_r = aa[24], AU12_r = aa[25], AU14_r = aa[26], AU15_r = aa[27], AU17_r = aa[28], AU20_r = aa[29], AU23_r = aa[30],
			AU25_r = aa[31], AU26_r = aa[32], AU28_r = aa[33], AU45_r = aa[34];

		//удивление
		e1.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU05_c*AU05_r + AU26_c*AU26_r) / 4);
		
		e1.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU05_c*AU05_r) / 3);
		e1.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU26_c*AU26_r) / 3);
		
		e1.push_back((AU05_c*AU05_r + AU26_c*AU26_r) / 2);
		


		//%страх
		
		e2.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU04_c*AU04_r + AU05_c*AU05_r + AU25_c*AU25_r + AU26_c*AU26_r) / 6);
		
		e2.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU04_c*AU04_r + AU05_c*AU05_r) / 4);
		e2.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU05_c*AU05_r + AU25_c*AU25_r) / 4);
		e2.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU05_c*AU05_r + AU25_c*AU25_r + AU26_c*AU26_r) / 5);
		
		e2.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU05_c*AU05_r + AU26_c*AU26_r) / 4);
		
		e2.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU05_c*AU05_r) / 3);
		e2.push_back((AU05_c*AU05_r + AU20_c*AU20_r + AU25_c*AU25_r) / 3);
		e2.push_back((AU05_c*AU05_r + AU20_c*AU20_r + AU25_c*AU25_r + AU26_c*AU26_r) / 4);

		e2.push_back((AU05_c*AU05_r + AU20_c*AU20_r + AU26_c*AU26_r) / 3);
		
		e2.push_back((AU05_c*AU05_r + AU20_c*AU20_r) / 2);
		//%радость
		e3.push_back((AU06_c*AU06_r + AU12_c*AU12_r) / 2);
		e3.push_back((AU12_c*AU12_r));
		//%ечаль
		
		e4.push_back((AU01_c*AU01_r + AU04_c*AU04_r + AU15_c*AU15_r) / 3);
		e4.push_back((AU06_c*AU06_r + AU15_c*AU15_r) / 2);
	
		e4.push_back((AU01_c*AU01_r + AU04_c*AU04_r + AU15_c*AU15_r + AU17_c*AU17_r) / 4);
	
		//%отвращение
		e5.push_back((AU09_c*AU09_r));

		e5.push_back((AU09_c*AU09_r + AU17_c*AU17_r) / 2);
		e5.push_back((AU10_c*AU10_r));

		e5.push_back((AU10_c*AU10_r + AU17_c*AU17_r) / 2);
		//%гнев
		
		e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU10_c*AU10_r + AU23_c*AU23_r + AU25_c*AU25_r) / 6);
		e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU10_c*AU10_r + AU23_c*AU23_r + AU26_c*AU26_r) / 6);
		e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r + AU25_c*AU25_r) / 5);
		e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r + AU26_c*AU26_r) / 5);
		e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU17_c*AU17_r + AU23_c*AU23_r) / 5);
		
		e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r) / 4);
		
		e6.push_back((AU05_c*AU05_r + AU07_c*AU07_r + AU10_c*AU10_r + AU23_c*AU23_r + AU25_c*AU25_r) / 5);
		e6.push_back((AU05_c*AU05_r + AU07_c*AU07_r + AU10_c*AU10_r + AU23_c*AU23_r + AU26_c*AU26_r) / 5);
		e6.push_back((AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r + AU25_c*AU25_r) / 4);
		e6.push_back((AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r + AU26_c*AU26_r) / 4);
		e6.push_back((AU05_c*AU05_r + AU07_c*AU07_r + AU17_c*AU17_r + AU23_c*AU23_r) / 4);
		
		e6.push_back((AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r) / 3);
		
		e6.push_back((AU04_c*AU04_r + AU07_c*AU07_r + AU10_c*AU10_r + AU23_c*AU23_r + AU25_c*AU25_r) / 5);
		e6.push_back((AU04_c*AU04_r + AU07_c*AU07_r + AU10_c*AU10_r + AU23_c*AU23_r + AU26_c*AU26_r) / 5);
		e6.push_back((AU04_c*AU04_r + AU07_c*AU07_r + AU23_c*AU23_r + AU25_c*AU25_r) / 4);
		e6.push_back((AU04_c*AU04_r + AU07_c*AU07_r + AU23_c*AU23_r + AU26_c*AU26_r) / 4);
		e6.push_back((AU04_c*AU04_r + AU07_c*AU07_r + AU17_c*AU17_r + AU23_c*AU23_r) / 4);
	
		e6.push_back((AU04_c*AU04_r + AU07_c*AU07_r + AU23_c*AU23_r) / 3);
		
		e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU10_c*AU10_r + AU23_c*AU23_r + AU25_c*AU25_r) / 5);
		e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU10_c*AU10_r + AU23_c*AU23_r + AU26_c*AU26_r) / 5);
		e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU23_c*AU23_r + AU25_c*AU25_r) / 4);
		e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU23_c*AU23_r + AU26_c*AU26_r) / 4);
		e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU17_c*AU17_r + AU23_c*AU23_r) / 4);
		
		e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU23_c*AU23_r) / 3);
		

		
		e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r + AU25_c*AU25_r) / 5);
		e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r + AU26_c*AU26_r) / 5);

	



		std::vector<double>::iterator result1;
		result1 = std::max_element(e1.begin(), e1.end());
		int r1 = std::distance(e1.begin(), result1);

		std::vector<double>::iterator result2;
		result2 = std::max_element(e2.begin(), e2.end());
		int r2 = std::distance(e2.begin(), result2);

		std::vector<double>::iterator result3;
		result3 = std::max_element(e3.begin(), e3.end());
		int r3 = std::distance(e3.begin(), result3);

		std::vector<double>::iterator result4;
		result4 = std::max_element(e4.begin(), e4.end());
		int r4 = std::distance(e4.begin(), result4);

		std::vector<double>::iterator result5;
		result5 = std::max_element(e5.begin(), e5.end());
		int r5 = std::distance(e5.begin(), result5);

		std::vector<double>::iterator result6;
		result6 = std::max_element(e6.begin(), e6.end());
		int r6 = std::distance(e6.begin(), result6);

		std::vector<double> em;
		em.push_back(e1[r1]);
		em.push_back(e2[r2]);
		em.push_back(e3[r3]);
		em.push_back(e4[r4]);
		em.push_back(e5[r5]);
		em.push_back(e6[r6]);


		for (int i = 0; i < em.size(); i++)
		{
			std::cout << em[i] << std::endl;
		}

		std::vector<double>::iterator result;
		result = std::max_element(em.begin(), em.end());
		int r = std::distance(em.begin(), result);

		double a = em[r];

		double sum_of_elems = std::accumulate(em.begin(), em.end(), 0);

		string strr, strr2;
		if (sum_of_elems == 0)
		{
			outfile << " " << "neutral 100%";

			strr = "neutral";

		}

	
		else
		{

			if (r == 0)
			{
				outfile << " " << "surprise";
				strr = "surprise";
			}
			if (r == 1)
			{
				outfile << " " << "fear";
				strr = "fear";
			}
			if (r == 2)
			{
				outfile << " " << "happiness";
				strr = "happiness";
			}
			if (r == 3)
			{
				outfile << " " << "sad";
				strr = "sad";
			}
			if (r == 4)
			{
				outfile << " " << "disgust";
				strr = "disgust";
			}
			if (r == 5)
			{
				outfile << " " << "anger";
				strr = "anger";
			}
	
			em[r] = 0;
			std::vector<double>::iterator result7;
			result7 = std::max_element(em.begin(), em.end());
			int r7 = std::distance(em.begin(), result7);
			double a2 = em[r7];
			if (r7 == 0)
			{
				outfile << " " << "surprise";
				strr2 = "surprise";
			}
			if (r7 == 1)
			{
				outfile << " " << "fear";
				strr2 = "fear";
			}
			if (r7 == 2)
			{
				outfile << " " << "happiness";
				strr2 = "happiness";
			}
			if (r7 == 3)
			{
				outfile << " " << "sad";
				strr2 = "sad";
			}
			if (r7 == 4)
			{
				outfile << " " << "disgust";
				strr2 = "disgust";
			}
			if (r7 == 5)
			{
				outfile << " " << "anger";
				strr2 = "anger";
			}

			double sum = a + a2;
			double z1 = (a / sum) * 100;
			double z2 = (a2 / sum) * 100;
			outfile << " " << z1 << "%";
			outfile << " " << z2 << "%";
	

		}
		aa.clear();
		outfile << endl;
	}

}

void prepareOutputFile(std::ofstream* output_file, bool output_2D_landmarks, bool output_3D_landmarks,
	bool output_model_params, bool output_pose, bool output_AUs, bool output_gaze,
	int num_landmarks, int num_model_modes, vector<string> au_names_class, vector<string> au_names_reg)
{

	*output_file << "frame, timestamp, confidence, success";

	if (output_gaze)
	{
		*output_file << ", gaze_0_x, gaze_0_y, gaze_0_z, gaze_1_x, gaze_1_y, gaze_2_z";
	}

	if (output_pose)
	{
		*output_file << ", pose_Tx, pose_Ty, pose_Tz, pose_Rx, pose_Ry, pose_Rz";
	}

	if (output_2D_landmarks)
	{
		for (int i = 0; i < num_landmarks; ++i)
		{
			*output_file << ", x_" << i;
		}
		for (int i = 0; i < num_landmarks; ++i)
		{
			*output_file << ", y_" << i;
		}
	}

	if (output_3D_landmarks)
	{
		for (int i = 0; i < num_landmarks; ++i)
		{
			*output_file << ", X_" << i;
		}
		for (int i = 0; i < num_landmarks; ++i)
		{
			*output_file << ", Y_" << i;
		}
		for (int i = 0; i < num_landmarks; ++i)
		{
			*output_file << ", Z_" << i;
		}
	}

	// Outputting model parameters (rigid and non-rigid), the first parameters are the 6 rigid shape parameters, they are followed by the non rigid shape parameters
	if (output_model_params)
	{
		*output_file << ", p_scale, p_rx, p_ry, p_rz, p_tx, p_ty";
		for (int i = 0; i < num_model_modes; ++i)
		{
			*output_file << ", p_" << i;
		}
	}

	if (output_AUs)
	{
		std::sort(au_names_reg.begin(), au_names_reg.end());
		for (string reg_name : au_names_reg)
		{
			*output_file << ", " << reg_name << "_r";
		}

		std::sort(au_names_class.begin(), au_names_class.end());
		for (string class_name : au_names_class)
		{
			*output_file << ", " << class_name << "_c";
		}
	}

	*output_file << endl;

}

// Output all of the information into one file in one go (quite a few parameters, but simplifies the flow)
void outputAllFeatures(std::ofstream* output_file, bool output_2D_landmarks, bool output_3D_landmarks,
	bool output_model_params, bool output_pose, bool output_AUs, bool output_gaze,
	const LandmarkDetector::CLNF& face_model, int frame_count, double time_stamp, bool detection_success,
	cv::Point3f gazeDirection0, cv::Point3f gazeDirection1, const cv::Vec6d& pose_estimate, double fx, double fy, double cx, double cy,
	const FaceAnalysis::FaceAnalyser& face_analyser, cv::Mat& captured_image,std::vector<string> &vec)
{
	std::vector<string> vect;
	double confidence = 0.5 * (1 - face_model.detection_certainty);

	*output_file << frame_count + 1 << ", " << time_stamp << ", " << confidence << ", " << detection_success;

	// Output the estimated gaze
	if (output_gaze)
	{
		*output_file << ", " << gazeDirection0.x << ", " << gazeDirection0.y << ", " << gazeDirection0.z
			<< ", " << gazeDirection1.x << ", " << gazeDirection1.y << ", " << gazeDirection1.z;
	}

	// Output the estimated head pose
	if (output_pose)
	{
		if (face_model.tracking_initialised)
		{
			*output_file << ", " << pose_estimate[0] << ", " << pose_estimate[1] << ", " << pose_estimate[2]
				<< ", " << pose_estimate[3] << ", " << pose_estimate[4] << ", " << pose_estimate[5];
		}
		else
		{
			*output_file << ", 0, 0, 0, 0, 0, 0";
		}
	}

	// Output the detected 2D facial landmarks
	if (output_2D_landmarks)
	{
		for (int i = 0; i < face_model.pdm.NumberOfPoints() * 2; ++i)
		{
			if (face_model.tracking_initialised)
			{
				*output_file << ", " << face_model.detected_landmarks.at<double>(i);
			}
			else
			{
				*output_file << ", 0";
			}
		}
	}

	// Output the detected 3D facial landmarks
	if (output_3D_landmarks)
	{
		cv::Mat_<double> shape_3D = face_model.GetShape(fx, fy, cx, cy);
		for (int i = 0; i < face_model.pdm.NumberOfPoints() * 3; ++i)
		{
			if (face_model.tracking_initialised)
			{
				*output_file << ", " << shape_3D.at<double>(i);
			}
			else
			{
				*output_file << ", 0";
			}
		}
	}

	if (output_model_params)
	{
		for (int i = 0; i < 6; ++i)
		{
			if (face_model.tracking_initialised)
			{
				*output_file << ", " << face_model.params_global[i];
			}
			else
			{
				*output_file << ", 0";
			}
		}
		for (int i = 0; i < face_model.pdm.NumberOfModes(); ++i)
		{
			if (face_model.tracking_initialised)
			{
				*output_file << ", " << face_model.params_local.at<double>(i, 0);
			}
			else
			{
				*output_file << ", 0";
			}
		}
	}
	//vector<string> f = { "dsnfnmdf" }, m = { "jfdfmn" };
	string an, an2, bb, cc;
	string zz1, zz2;
	vector<double> aa;
	if (output_AUs)
	{
		
		auto aus_reg = face_analyser.GetCurrentAUsReg();

		vector<string> au_reg_names = face_analyser.GetAURegNames();
		std::sort(au_reg_names.begin(), au_reg_names.end());

		// write out ar the correct index
		for (string au_name : au_reg_names)
		{
			for (auto au_reg : aus_reg)
			{
				if (au_name.compare(au_reg.first) == 0)
				{
					if (au_reg.second < 0)
					{
						au_reg.second = 0;
					}
					aa.push_back(au_reg.second);
					//*output_file << ", " << au_reg.second;
					
					break;
				}
			}
		}

		if (aus_reg.size() == 0)
		{
			for (size_t p = 0; p < face_analyser.GetAURegNames().size(); ++p)
			{
				*output_file << ", 0";
			}
		}

		auto aus_class = face_analyser.GetCurrentAUsClass();

		vector<string> au_class_names = face_analyser.GetAUClassNames();
		std::sort(au_class_names.begin(), au_class_names.end());

		// write out ar the correct index
		for (string au_name : au_class_names)
		{
			for (auto au_class : aus_class)
			{
				if (au_name.compare(au_class.first) == 0)
				{
					aa.push_back(au_class.second);
					//*output_file << ", " << au_class.second;
					break;
				}
			}
		}

		if (aus_class.size() == 0)
		{
			for (size_t p = 0; p < face_analyser.GetAUClassNames().size(); ++p)
			{
				*output_file << ", 0";
			}
		}
	}


	vector<double> e1, e2, e3, e4, e5, e6;

	double AU01_c = aa[0], AU02_c = aa[1], AU04_c = aa[2], AU05_c = aa[3], AU06_c = aa[4], AU07_c = aa[5], AU09_c = aa[6],
		AU10_c = aa[7], AU12_c = aa[8], AU14_c = aa[9], AU15_c = aa[10], AU17_c = aa[11], AU20_c = aa[12], AU23_c = aa[13],
		AU25_c = aa[14], AU26_c = aa[15], AU45_c = aa[16];
	double AU01_r = aa[17], AU02_r = aa[18], AU04_r = aa[19], AU05_r = aa[20], AU06_r = aa[21], AU07_r = aa[22], AU09_r = aa[23],
		AU10_r = aa[24], AU12_r = aa[25], AU14_r = aa[26], AU15_r = aa[27], AU17_r = aa[28], AU20_r = aa[29], AU23_r = aa[30],
		AU25_r = aa[31], AU26_r = aa[32], AU28_r = aa[33], AU45_r = aa[34];

	//удивление
	e1.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU05_c*AU05_r + AU26_c*AU26_r) / 4);
	
	e1.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU05_c*AU05_r) / 3);
	e1.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU26_c*AU26_r) / 3);

	e1.push_back((AU05_c*AU05_r + AU26_c*AU26_r) / 2);



	//%страх
	
	e2.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU04_c*AU04_r + AU05_c*AU05_r + AU25_c*AU25_r + AU26_c*AU26_r) / 6);
	
	e2.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU04_c*AU04_r + AU05_c*AU05_r) / 4);
	e2.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU05_c*AU05_r + AU25_c*AU25_r) / 4);
	e2.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU05_c*AU05_r + AU25_c*AU25_r + AU26_c*AU26_r) / 5);
	
	e2.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU05_c*AU05_r + AU26_c*AU26_r) / 4);

	e2.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU05_c*AU05_r) / 3);
	e2.push_back((AU05_c*AU05_r + AU20_c*AU20_r + AU25_c*AU25_r) / 3);
	e2.push_back((AU05_c*AU05_r + AU20_c*AU20_r + AU25_c*AU25_r + AU26_c*AU26_r) / 4);

	e2.push_back((AU05_c*AU05_r + AU20_c*AU20_r + AU26_c*AU26_r) / 3);

	e2.push_back((AU05_c*AU05_r + AU20_c*AU20_r) / 2);
	//%радость
	e3.push_back((AU06_c*AU06_r + AU12_c*AU12_r) / 2);
	e3.push_back((AU12_c*AU12_r));
	//%ечаль

	e4.push_back((AU01_c*AU01_r + AU04_c*AU04_r + AU15_c*AU15_r) / 3);
	e4.push_back((AU06_c*AU06_r + AU15_c*AU15_r) / 2);

	e4.push_back((AU01_c*AU01_r + AU04_c*AU04_r + AU15_c*AU15_r + AU17_c*AU17_r) / 4);

	//%отвращение
	e5.push_back((AU09_c*AU09_r));

	e5.push_back((AU09_c*AU09_r + AU17_c*AU17_r) / 2);
	e5.push_back((AU10_c*AU10_r));

	e5.push_back((AU10_c*AU10_r + AU17_c*AU17_r) / 2);
	//%гнев
	
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU10_c*AU10_r + AU23_c*AU23_r + AU25_c*AU25_r) / 6);
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU10_c*AU10_r + AU23_c*AU23_r + AU26_c*AU26_r) / 6);
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r + AU25_c*AU25_r) / 5);
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r + AU26_c*AU26_r) / 5);
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU17_c*AU17_r + AU23_c*AU23_r) / 5);
	
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r) / 4);


	
	e6.push_back((AU05_c*AU05_r + AU07_c*AU07_r + AU10_c*AU10_r + AU23_c*AU23_r + AU25_c*AU25_r) / 5);
	e6.push_back((AU05_c*AU05_r + AU07_c*AU07_r + AU10_c*AU10_r + AU23_c*AU23_r + AU26_c*AU26_r) / 5);
	e6.push_back((AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r + AU25_c*AU25_r) / 4);
	e6.push_back((AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r + AU26_c*AU26_r) / 4);
	e6.push_back((AU05_c*AU05_r + AU07_c*AU07_r + AU17_c*AU17_r + AU23_c*AU23_r) / 4);

	e6.push_back((AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r) / 3);



	e6.push_back((AU04_c*AU04_r + AU07_c*AU07_r + AU10_c*AU10_r + AU23_c*AU23_r + AU25_c*AU25_r) / 5);
	e6.push_back((AU04_c*AU04_r + AU07_c*AU07_r + AU10_c*AU10_r + AU23_c*AU23_r + AU26_c*AU26_r) / 5);
	e6.push_back((AU04_c*AU04_r + AU07_c*AU07_r + AU23_c*AU23_r + AU25_c*AU25_r) / 4);
	e6.push_back((AU04_c*AU04_r + AU07_c*AU07_r + AU23_c*AU23_r + AU26_c*AU26_r) / 4);
	e6.push_back((AU04_c*AU04_r + AU07_c*AU07_r + AU17_c*AU17_r + AU23_c*AU23_r) / 4);

	e6.push_back((AU04_c*AU04_r + AU07_c*AU07_r + AU23_c*AU23_r) / 3);

	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU10_c*AU10_r + AU23_c*AU23_r + AU25_c*AU25_r) / 5);
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU10_c*AU10_r + AU23_c*AU23_r + AU26_c*AU26_r) / 5);
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU23_c*AU23_r + AU25_c*AU25_r) / 4);
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU23_c*AU23_r + AU26_c*AU26_r) / 4);
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU17_c*AU17_r + AU23_c*AU23_r) / 4);

	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU23_c*AU23_r) / 3);

	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r + AU25_c*AU25_r) / 5);
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r + AU26_c*AU26_r) / 5);




	std::vector<double>::iterator result1;
	result1 = std::max_element(e1.begin(), e1.end());
	int r1 = std::distance(e1.begin(), result1);

	std::vector<double>::iterator result2;
	result2 = std::max_element(e2.begin(), e2.end());
	int r2 = std::distance(e2.begin(), result2);

	std::vector<double>::iterator result3;
	result3 = std::max_element(e3.begin(), e3.end());
	int r3 = std::distance(e3.begin(), result3);

	std::vector<double>::iterator result4;
	result4 = std::max_element(e4.begin(), e4.end());
	int r4 = std::distance(e4.begin(), result4);

	std::vector<double>::iterator result5;
	result5 = std::max_element(e5.begin(), e5.end());
	int r5 = std::distance(e5.begin(), result5);

	std::vector<double>::iterator result6;
	result6 = std::max_element(e6.begin(), e6.end());
	int r6 = std::distance(e6.begin(), result6);

	std::vector<double> em;
	em.push_back(e1[r1]);
	em.push_back(e2[r2]);
	em.push_back(e3[r3]);
	em.push_back(e4[r4]);
	em.push_back(e5[r5]);
	em.push_back(e6[r6]);


	for (int i = 0; i < em.size(); i++)
	{
		std::cout << em[i] << std::endl;
	}

	std::vector<double>::iterator result;
	result = std::max_element(em.begin(), em.end());
	int r = std::distance(em.begin(), result);

	double a = em[r];


	double sum_of_elems = std::accumulate(em.begin(), em.end(), 0);

	string strr, strr2;
	if (sum_of_elems == 0)
	{
		*output_file << " " << "neutral 100%";

		strr = "neutral";

	}


	else
	{

		if (r == 0)
		{
			*output_file << " " << "surprise";
			strr = "surprise";
		}
		if (r == 1)
		{
			*output_file << " " << "fear";
			strr = "fear";
		}
		if (r == 2)
		{
			*output_file << " " << "happiness";
			strr = "happiness";
		}
		if (r == 3)
		{
			*output_file << " " << "sad";
			strr = "sad";
		}
		if (r == 4)
		{
			*output_file << " " << "disgust";
			strr = "disgust";
		}
		if (r == 5)
		{
			*output_file << " " << "anger";
			strr = "anger";
		}

		em[r] = 0;
		std::vector<double>::iterator result7;
		result7 = std::max_element(em.begin(), em.end());
		int r7 = std::distance(em.begin(), result7);
		double a2 = em[r7];
		if (r7 == 0)
		{
			*output_file << " " << "surprise";
			strr2 = "surprise";
		}
		if (r7 == 1)
		{
			*output_file << " " << "fear";
			strr2 = "fear";
		}
		if (r7 == 2)
		{
			*output_file << " " << "happiness";
			strr2 = "happiness";
		}
		if (r7 == 3)
		{
			*output_file << " " << "sad";
			strr2 = "sad";
		}
		if (r7 == 4)
		{
			*output_file << " " << "disgust";
			strr2 = "disgust";
		}
		if (r7 == 5)
		{
			*output_file << " " << "anger";
			strr2 = "anger";
		}

		double sum = a + a2;
		double z1 = (a / sum) * 100;
		double z2 = (a2 / sum) * 100;
		*output_file << " " << z1 << "%";
		*output_file << " " << z2 << "%";

		
		zz1= to_string(z1) + "%";
		
		zz2= to_string(z2) + "%";


	}
	aa.clear();

	*output_file << endl;

	
		vect.push_back(strr);
		vect.push_back(strr2);
		vect.push_back(zz1);
		vect.push_back(zz2);
	
	return vec.swap(vect);
		
	
}


void get_output_feature_params(vector<string> &output_similarity_aligned, vector<string> &output_hog_aligned_files, double &similarity_scale,
	int &similarity_size, bool &grayscale, bool& verbose, bool& dynamic,
	bool &output_2D_landmarks, bool &output_3D_landmarks, bool &output_model_params, bool &output_pose, bool &output_AUs, bool &output_gaze,
	vector<string> &arguments)
{
	output_similarity_aligned.clear();
	output_hog_aligned_files.clear();

	bool* valid = new bool[arguments.size()];

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;
	}

	string output_root = "";

	// By default the model is dynamic
	dynamic = true;

	string separator = string(1, boost::filesystem::path::preferred_separator);

	// First check if there is a root argument (so that videos and outputs could be defined more easilly)
	for (size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-root") == 0)
		{
			output_root = arguments[i + 1] + separator;
			i++;
		}
		if (arguments[i].compare("-outroot") == 0)
		{
			output_root = arguments[i + 1] + separator;
			i++;
		}
	}

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-simalign") == 0)
		{
			output_similarity_aligned.push_back(output_root + arguments[i + 1]);
			create_directory(output_root + arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-hogalign") == 0)
		{
			output_hog_aligned_files.push_back(output_root + arguments[i + 1]);
			create_directory_from_file(output_root + arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-verbose") == 0)
		{
			verbose = true;
		}
		else if (arguments[i].compare("-au_static") == 0)
		{
			dynamic = false;
		}
		else if (arguments[i].compare("-g") == 0)
		{
			grayscale = true;
			valid[i] = false;
		}
		else if (arguments[i].compare("-simscale") == 0)
		{
			similarity_scale = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-simsize") == 0)
		{
			similarity_size = stoi(arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-no2Dfp") == 0)
		{
			output_2D_landmarks = false;
			valid[i] = false;
		}
		else if (arguments[i].compare("-no3Dfp") == 0)
		{
			output_3D_landmarks = false;
			valid[i] = false;
		}
		else if (arguments[i].compare("-noMparams") == 0)
		{
			output_model_params = false;
			valid[i] = false;
		}
		else if (arguments[i].compare("-noPose") == 0)
		{
			output_pose = false;
			valid[i] = false;
		}
		else if (arguments[i].compare("-noAUs") == 0)
		{
			output_AUs = false;
			valid[i] = false;
		}
		else if (arguments[i].compare("-noGaze") == 0)
		{
			output_gaze = false;
			valid[i] = false;
		}
	}

	for (int i = arguments.size() - 1; i >= 0; --i)
	{
		if (!valid[i])
		{
			arguments.erase(arguments.begin() + i);
		}
	}

}

// Can process images via directories creating a separate output file per directory
void get_image_input_output_params_feats(vector<vector<string> > &input_image_files, bool& as_video, vector<string> &arguments)
{
	bool* valid = new bool[arguments.size()];

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;
		if (arguments[i].compare("-fdir") == 0)
		{

			// parse the -fdir directory by reading in all of the .png and .jpg files in it
			path image_directory(arguments[i + 1]);

			try
			{
				// does the file exist and is it a directory
				if (exists(image_directory) && is_directory(image_directory))
				{

					vector<path> file_in_directory;
					copy(directory_iterator(image_directory), directory_iterator(), back_inserter(file_in_directory));

					// Sort the images in the directory first
					sort(file_in_directory.begin(), file_in_directory.end());

					vector<string> curr_dir_files;

					for (vector<path>::const_iterator file_iterator(file_in_directory.begin()); file_iterator != file_in_directory.end(); ++file_iterator)
					{
						// Possible image extension .jpg and .png
						if (file_iterator->extension().string().compare(".jpg") == 0 || file_iterator->extension().string().compare(".png") == 0)
						{
							curr_dir_files.push_back(file_iterator->string());
						}
					}

					input_image_files.push_back(curr_dir_files);
				}
			}
			catch (const filesystem_error& ex)
			{
				cout << ex.what() << '\n';
			}

			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-asvid") == 0)
		{
			as_video = true;
		}
	}

	// Clear up the argument list
	for (int i = arguments.size() - 1; i >= 0; --i)
	{
		if (!valid[i])
		{
			arguments.erase(arguments.begin() + i);
		}
	}

}

void output_HOG_frame(std::ofstream* hog_file, bool good_frame, const cv::Mat_<double>& hog_descriptor, int num_rows, int num_cols)
{

	// Using FHOGs, hence 31 channels
	int num_channels = 31;

	hog_file->write((char*)(&num_cols), 4);
	hog_file->write((char*)(&num_rows), 4);
	hog_file->write((char*)(&num_channels), 4);

	// Not the best way to store a bool, but will be much easier to read it
	float good_frame_float;
	if (good_frame)
		good_frame_float = 1;
	else
		good_frame_float = -1;

	hog_file->write((char*)(&good_frame_float), 4);

	cv::MatConstIterator_<double> descriptor_it = hog_descriptor.begin();

	for (int y = 0; y < num_cols; ++y)
	{
		for (int x = 0; x < num_rows; ++x)
		{
			for (unsigned int o = 0; o < 31; ++o)
			{

				float hog_data = (float)(*descriptor_it++);
				hog_file->write((char*)&hog_data, 4);
			}
		}
	}
}

void outputAllF(const FaceAnalysis::FaceAnalyser& face_analyser, cv::Mat& captured_image, std::vector<string> &vec)
{
	std::cout << "function AU" << std::endl;
	std::vector<string> vect;

	string an, an2, bb, cc;
	string zz1, zz2;
	vector<double> aa;


	auto aus_reg = face_analyser.GetCurrentAUsReg();

	vector<string> au_reg_names = face_analyser.GetAURegNames();
	std::sort(au_reg_names.begin(), au_reg_names.end());

	// write out ar the correct index
	for (string au_name : au_reg_names)
	{
		for (auto au_reg : aus_reg)
		{
			if (au_name.compare(au_reg.first) == 0)
			{
				if (au_reg.second < 0)
				{
					au_reg.second = 0;
				}
				aa.push_back(au_reg.second);
				//*output_file << ", " << au_reg.second;
				std::cout << au_reg.second << std::endl;

				break;
			}
		}
	}



	auto aus_class = face_analyser.GetCurrentAUsClass();

	vector<string> au_class_names = face_analyser.GetAUClassNames();
	std::sort(au_class_names.begin(), au_class_names.end());

	// write out ar the correct index
	for (string au_name : au_class_names)
	{
		for (auto au_class : aus_class)
		{
			if (au_name.compare(au_class.first) == 0)
			{
				aa.push_back(au_class.second);
				//*output_file << ", " << au_class.second;
				break;
			}
		}
	}
	std::cout << aa.size() << std::endl;
	for (int i = 0; i < aa.size(); i++)
	{
		std::cout << aa[i] << std::endl;
	}


	std::cout << "function AU_2" << std::endl;

	vector<double> e1, e2, e3, e4, e5, e6;

	double AU01_c = aa[0], AU02_c = aa[1], AU04_c = aa[2], AU05_c = aa[3], AU06_c = aa[4], AU07_c = aa[5], AU09_c = aa[6],
		AU10_c = aa[7], AU12_c = aa[8], AU14_c = aa[9], AU15_c = aa[10], AU17_c = aa[11], AU20_c = aa[12], AU23_c = aa[13],
		AU25_c = aa[14], AU26_c = aa[15], AU45_c = aa[16];
	double AU01_r = aa[17], AU02_r = aa[18], AU04_r = aa[19], AU05_r = aa[20], AU06_r = aa[21], AU07_r = aa[22], AU09_r = aa[23],
		AU10_r = aa[24], AU12_r = aa[25], AU14_r = aa[26], AU15_r = aa[27], AU17_r = aa[28], AU20_r = aa[29], AU23_r = aa[30],
		AU25_r = aa[31], AU26_r = aa[32], AU28_r = aa[33], AU45_r = aa[34];

	//удивление
	e1.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU05_c*AU05_r + AU26_c*AU26_r) / 4);

	e1.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU05_c*AU05_r) / 3);
	e1.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU26_c*AU26_r) / 3);

	e1.push_back((AU05_c*AU05_r + AU26_c*AU26_r) / 2);



	//%страх
	
	e2.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU04_c*AU04_r + AU05_c*AU05_r + AU25_c*AU25_r + AU26_c*AU26_r) / 6);

	e2.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU04_c*AU04_r + AU05_c*AU05_r) / 4);
	e2.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU05_c*AU05_r + AU25_c*AU25_r) / 4);
	e2.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU05_c*AU05_r + AU25_c*AU25_r + AU26_c*AU26_r) / 5);
	
	e2.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU05_c*AU05_r + AU26_c*AU26_r) / 4);
	
	e2.push_back((AU01_c*AU01_r + AU02_c*AU02_r + AU05_c*AU05_r) / 3);
	e2.push_back((AU05_c*AU05_r + AU20_c*AU20_r + AU25_c*AU25_r) / 3);
	e2.push_back((AU05_c*AU05_r + AU20_c*AU20_r + AU25_c*AU25_r + AU26_c*AU26_r) / 4);

	e2.push_back((AU05_c*AU05_r + AU20_c*AU20_r + AU26_c*AU26_r) / 3);

	e2.push_back((AU05_c*AU05_r + AU20_c*AU20_r) / 2);
	//%радость
	e3.push_back((AU06_c*AU06_r + AU12_c*AU12_r) / 2);
	e3.push_back((AU12_c*AU12_r));
	//%ечаль

	e4.push_back((AU01_c*AU01_r + AU04_c*AU04_r + AU15_c*AU15_r) / 3);
	e4.push_back((AU06_c*AU06_r + AU15_c*AU15_r) / 2);

	e4.push_back((AU01_c*AU01_r + AU04_c*AU04_r + AU15_c*AU15_r + AU17_c*AU17_r) / 4);

	//%отвращение
	e5.push_back((AU09_c*AU09_r));

	e5.push_back((AU09_c*AU09_r + AU17_c*AU17_r) / 2);
	e5.push_back((AU10_c*AU10_r));

	e5.push_back((AU10_c*AU10_r + AU17_c*AU17_r) / 2);
	//%гнев
	
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU10_c*AU10_r + AU23_c*AU23_r + AU25_c*AU25_r) / 6);
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU10_c*AU10_r + AU23_c*AU23_r + AU26_c*AU26_r) / 6);
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r + AU25_c*AU25_r) / 5);
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r + AU26_c*AU26_r) / 5);
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU17_c*AU17_r + AU23_c*AU23_r) / 5);

	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r) / 4);


	
	e6.push_back((AU05_c*AU05_r + AU07_c*AU07_r + AU10_c*AU10_r + AU23_c*AU23_r + AU25_c*AU25_r) / 5);
	e6.push_back((AU05_c*AU05_r + AU07_c*AU07_r + AU10_c*AU10_r + AU23_c*AU23_r + AU26_c*AU26_r) / 5);
	e6.push_back((AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r + AU25_c*AU25_r) / 4);
	e6.push_back((AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r + AU26_c*AU26_r) / 4);
	e6.push_back((AU05_c*AU05_r + AU07_c*AU07_r + AU17_c*AU17_r + AU23_c*AU23_r) / 4);

	e6.push_back((AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r) / 3);


	
	e6.push_back((AU04_c*AU04_r + AU07_c*AU07_r + AU10_c*AU10_r + AU23_c*AU23_r + AU25_c*AU25_r) / 5);
	e6.push_back((AU04_c*AU04_r + AU07_c*AU07_r + AU10_c*AU10_r + AU23_c*AU23_r + AU26_c*AU26_r) / 5);
	e6.push_back((AU04_c*AU04_r + AU07_c*AU07_r + AU23_c*AU23_r + AU25_c*AU25_r) / 4);
	e6.push_back((AU04_c*AU04_r + AU07_c*AU07_r + AU23_c*AU23_r + AU26_c*AU26_r) / 4);
	e6.push_back((AU04_c*AU04_r + AU07_c*AU07_r + AU17_c*AU17_r + AU23_c*AU23_r) / 4);

	e6.push_back((AU04_c*AU04_r + AU07_c*AU07_r + AU23_c*AU23_r) / 3);


	
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU10_c*AU10_r + AU23_c*AU23_r + AU25_c*AU25_r) / 5);
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU10_c*AU10_r + AU23_c*AU23_r + AU26_c*AU26_r) / 5);
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU23_c*AU23_r + AU25_c*AU25_r) / 4);
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU23_c*AU23_r + AU26_c*AU26_r) / 4);
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU17_c*AU17_r + AU23_c*AU23_r) / 4);
	
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU23_c*AU23_r) / 3);

	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r + AU25_c*AU25_r) / 5);
	e6.push_back((AU04_c*AU04_r + AU05_c*AU05_r + AU07_c*AU07_r + AU23_c*AU23_r + AU26_c*AU26_r) / 5);





	std::vector<double>::iterator result1;
	result1 = std::max_element(e1.begin(), e1.end());
	int r1 = std::distance(e1.begin(), result1);

	std::vector<double>::iterator result2;
	result2 = std::max_element(e2.begin(), e2.end());
	int r2 = std::distance(e2.begin(), result2);

	std::vector<double>::iterator result3;
	result3 = std::max_element(e3.begin(), e3.end());
	int r3 = std::distance(e3.begin(), result3);

	std::vector<double>::iterator result4;
	result4 = std::max_element(e4.begin(), e4.end());
	int r4 = std::distance(e4.begin(), result4);

	std::vector<double>::iterator result5;
	result5 = std::max_element(e5.begin(), e5.end());
	int r5 = std::distance(e5.begin(), result5);

	std::vector<double>::iterator result6;
	result6 = std::max_element(e6.begin(), e6.end());
	int r6 = std::distance(e6.begin(), result6);

	std::vector<double> em;
	em.push_back(e1[r1]);
	em.push_back(e2[r2]);
	em.push_back(e3[r3]);
	em.push_back(e4[r4]);
	em.push_back(e5[r5]);
	em.push_back(e6[r6]);
	std::cout << "function AU_3" << std::endl;

	for (int i = 0; i < em.size(); i++)
	{
		std::cout << em[i] << std::endl;
	}

	std::vector<double>::iterator result;
	result = std::max_element(em.begin(), em.end());
	int r = std::distance(em.begin(), result);

	double a = em[r];

	double sum_of_elems = std::accumulate(em.begin(), em.end(), 0);

	string strr, strr2;
	if (sum_of_elems == 0)
	{


		strr = "neutral";

	}


	else
	{

		if (r == 0)
		{
		
			strr = "surprise";
		}
		if (r == 1)
		{

			strr = "fear";
		}
		if (r == 2)
		{
		
			strr = "happiness";
		}
		if (r == 3)
		{
		
			strr = "sad";
		}
		if (r == 4)
		{
		
			strr = "disgust";
		}
		if (r == 5)
		{
	
			strr = "anger";
		}

		em[r] = 0;
		std::vector<double>::iterator result7;
		result7 = std::max_element(em.begin(), em.end());
		int r7 = std::distance(em.begin(), result7);
		double a2 = em[r7];
		if (r7 == 0)
		{
			
			strr2 = "surprise";
		}
		if (r7 == 1)
		{
			
			strr2 = "fear";
		}
		if (r7 == 2)
		{
			
			strr2 = "happiness";
		}
		if (r7 == 3)
		{
		
			strr2 = "sad";
		}
		if (r7 == 4)
		{
		
			strr2 = "disgust";
		}
		if (r7 == 5)
		{
		
			strr2 = "anger";
		}

		double sum = a + a2;
		double z1 = (a / sum) * 100;
		double z2 = (a2 / sum) * 100;



		zz1 = to_string(z1) + "%";

		zz2 = to_string(z2) + "%";


	}
	aa.clear();


	vect.push_back(strr);
	vect.push_back(strr2);
	vect.push_back(zz1);
	vect.push_back(zz2);

	return vec.swap(vect);


}

#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <stdio.h>
#include <winsock2.h> // Wincosk2.h ������ ���� ������ windows!
#include <cstdlib>
#include <iostream>
#include <winsock.h>
// Need to link with Ws2_32.lib
#pragma comment (lib, "Ws2_32.lib")
// #pragma comment (lib, "Mswsock.lib")

#include <vector>
// OpenCV includes
#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
int main() {
	setlocale(LC_ALL, "rus");
	char url[100];
	int port;
	char my_name[30] = "ASA";
	/*std::cout << "Url: ";
	cin >> url;
	cout << endl;
	*/
	std::cout << "Port: ";
	cin >> port;
	cout << endl;

	WSADATA WsaData;
	if (int err = WSAStartup(MAKEWORD(2, 0), &WsaData) != 0)
	{
		std::cout << "Socket not Loaded!\n";
	}
	else {
		std::cout << "Socket Loaded  \n";
	}

	int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (sock == -1) {
		std::cout << "Error! Socket no created.\n";
	}
	else {
		std::cout << "Socket Create.\n";
	}

	sockaddr_in addr;
	addr.sin_family = AF_INET;
	addr.sin_port = htons(port);
	//addr.sin_addr.s_addr = inet_addr(url);
	addr.sin_addr.s_addr = inet_addr("127.0.0.1");
	int locate;
	locate = connect(sock, (sockaddr *)&addr, sizeof(addr));

	if (locate < 0) {

		std::cout << "Fatal Error!\n";
		//system("pause");
	}
	//////////////
	/*
	char cut[10000];
	//cout << get << endl;
	std::cout << "Enter get: ";
	std::cin >> cut;
	send(sock, cut, 10000, 0);
	*/
	////////////////
	/*else {
		char cut[10000];
		char get[10000];
		send(sock, my_name, 30, 0);
		recv(sock, get, 10000, 0);
		cout << get << endl;
		std::cout << "Enter get: ";
		std::cin >> cut;

		send(sock, cut, 10000, 0);
		*/

	VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.
	int count = 0;
	//unconditional loop
	while (true) {
		Mat cameraFrame;
		stream1.read(cameraFrame);
		imshow("Client_camera", cameraFrame);
		count++;
		printf("%d", count);
		if (waitKey(30) >= 0)
			break;
		if (count % 5==0)
		{
			//char *sendbuf = "this is a test";
			//printf("S<=C:");
			//fgets(&buff[0], sizeof(buff) - 1, stdin);
			// ��������� ����������
			Mat srcMat;                                                                 // ������ ��������� �����������
			vector<uchar> imgBuf;                                                // ����� ��� ������� �����������
			vector<int> quality_params = vector<int>(2);              // ������ ���������� �������� ������
			quality_params[0] = CV_IMWRITE_JPEG_QUALITY; // ����� JPEG
			quality_params[1] = 90;                                               // �� ��������� �������� ������ (95) 0-100

																				  // ����������� ���� �����
																				  //cd.frame = cvQueryFrame(cd.videocap);

																				  // �������� ����������� � ������
			srcMat = cameraFrame;

			// �������� ����������� ������� JPEG
			imencode(".jpg", srcMat, imgBuf, quality_params);
			// ���������� ������
			send(sock, (const char*)&imgBuf[0], static_cast<int>(imgBuf.size()), 0);
			//locate = connect(sock, (sockaddr *)&addr, sizeof(addr));
			//send(my_sock, sendbuf, (int)strlen(sendbuf), 0);
		}
		//system("pause");

	}
}
//}

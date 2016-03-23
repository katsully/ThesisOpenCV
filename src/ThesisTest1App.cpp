#include "cinder/app/App.h"
#include "cinder/params/Params.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

#include "Kinect2.h"
#include "CinderOpenCV.h"

#include "Shape.h"

using namespace ci;
using namespace ci::app;
using namespace std;

class ThesisTest1App : public App {
public:
	ThesisTest1App();

	void setup();
	void prepareSettings(Settings* settings);
	void update() override;
	void draw() override;

	// all pixels below near limit and above far limit are set to far limit depth
	short mNearLimit;
	short mFarLimit;

	// threshold for the camera
	double mThresh;
	double mMaxVal;
private:
	Kinect2::DeviceRef mDevice;
	ci::Channel8uRef mChannelBodyIndex;
	ci::Channel16uRef mChannelDepth;
	ci::Channel16uRef mChannelInfrared;
	ci::Surface8uRef mSurfaceColor;

	float mFrameRate;
	bool mFullScreen;
	ci::params::InterfaceGlRef mParams;

	cv::Mat mInput;

	ci::Surface8u mSurface;
	ci::Surface8u mSurfaceDepth;
	ci::Surface8u mSurfaceBlur;
	ci::Surface8u mSurfaceSubtract;
	gl::TextureRef mTexture;
	gl::TextureRef mTextureDepth;

	cv::Mat mPreviousFrame;
	cv::Mat mBackground;

	typedef vector< vector<cv::Point > > ContourVector;
	ContourVector mContours;
	ContourVector mApproxContours;
	int mStepSize;
	int mBlurAmount;
	int shapeUID;

	cv::vector<cv::Vec4i> mHierarchy;
	vector<Shape> mShapes;
	// store tracked shapes
	vector<Shape> mTrackedShapes;

	cv::Mat removeBlack(cv::Mat input, short nearLimit, short farLimit);
	vector< Shape > getEvaluationSet(ContourVector rawContours, int minimalArea, int maxArea);
	Shape* findNearestMatch(Shape trackedShape, vector< Shape > &shapes, float maximumDistance);
};

ThesisTest1App::ThesisTest1App()
{
	mFrameRate = 0.0f;
	mFullScreen = false;

	mDevice = Kinect2::Device::create();
	mDevice->start();
	mDevice->connectBodyIndexEventHandler([&](const Kinect2::BodyIndexFrame& frame)
	{
		mChannelBodyIndex = frame.getChannel();
	});
	mDevice->connectColorEventHandler([&](const Kinect2::ColorFrame& frame)
	{
		mSurfaceColor = frame.getSurface();
	});
	mDevice->connectDepthEventHandler([&](const Kinect2::DepthFrame& frame)
	{
		mChannelDepth = frame.getChannel();
	});
	mDevice->connectInfraredEventHandler([&](const Kinect2::InfraredFrame& frame)
	{
		mChannelInfrared = frame.getChannel();
	});

	mParams = params::InterfaceGl::create("Params", ivec2(255, 200));
	//mParams->addParam("Frame rate", &mFrameRate, "", true);
	//mParams->addParam("Full screen", &mFullScreen).key("f");
	//mParams->addButton("Quit", [&]() { quit(); }, "key=q");
	mParams->addParam("Thresh", &mThresh, "min=0.0f max=255.0f step 1.0 keyIncr=a keyDecr=s");
	mParams->addParam("Maxval", &mMaxVal, "min=0.0f max=255.0f step=1.0 keyIncr=q keyDecr=w");
	mStepSize = 10;
	mBlurAmount = 10;
}

void ThesisTest1App::prepareSettings(Settings* settings) {
	settings->setFrameRate(60.0f);
	settings->setWindowSize(800, 800);
}

void ThesisTest1App::setup()
{
	shapeUID = 0;

	// start off drawing shapes, not points
	//mDrawShapes = true;

	// set the threshold to ignore all black pixels and pixels that are far away from the camera
	mNearLimit = 30;
	mFarLimit = 4000;
	mThresh = 0.0;
	mMaxVal = 255.0;
}

void ThesisTest1App::update()
{
	if (mChannelDepth) {
		//Channel16u channel;
		//channel = *mChannelDepth;
		mInput = toOcv(mChannelDepth->clone(true));

		cv::Mat thresh;
		cv::Mat eightBit;
		cv::Mat withoutBlack;

		// remove black pixels from frame which get detected as noise
		withoutBlack = removeBlack(mInput, mNearLimit, mFarLimit);

		// convert matrix from 16 bit to 8 bit with some color compensation
		withoutBlack.convertTo(eightBit, CV_8UC3, 0.1 / 1.0);

		// invert the image
		cv::bitwise_not(eightBit, eightBit);

		mContours.clear();
		mApproxContours.clear();

		// using a threshold to reduce noise
		cv::threshold(eightBit, thresh, mThresh, mMaxVal, CV_8U);

		// draw lines around shapes
		cv::findContours(thresh, mContours, mHierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		vector<cv::Point> approx;
		// approx number of points per contour
		for (int i = 0; i < mContours.size(); i++) {
			cv::approxPolyDP(mContours[i], approx, 3, true);
			mApproxContours.push_back(approx);
		}

		mShapes.clear();
		// get data that we can later compare
		mShapes = getEvaluationSet(mApproxContours, 75, 100000);

		// find the nearest match for each shape
		for (int i = 0; i < mTrackedShapes.size(); i++) {
			Shape* nearestShape = findNearestMatch(mTrackedShapes[i], mShapes, 5000);

			// a tracked shape was found, update that tracked shape with the new shape
			if (nearestShape != NULL) {
				nearestShape->matchFound = true;
				mTrackedShapes[i].centroid = nearestShape->centroid;
				// get depth value from center point
				//float centerDepth = (float)mInput.at<short>(mTrackedShapes[i].centroid.y, mTrackedShapes[i].centroid.x);
				// map 10 4000 to 0 1
				//mTrackedShapes[i].depth = lmap(centerDepth, (float)mNearLimit, (float)mFarLimit, 0.0f, 1.0f);
				mTrackedShapes[i].lastFrameSeen = ci::app::getElapsedFrames();
				mTrackedShapes[i].hull.clear();
				mTrackedShapes[i].hull = nearestShape->hull;
				//mTrackedShapes[i].moving = nearestShape->moving;
				//mTrackedShapes[i].motion = nearestShape->motion;
			}
		}

		// if shape->matchFound is false, add it as a new shape
		for (int i = 0; i < mShapes.size(); i++) {
			if (mShapes[i].matchFound == false) {
				// assign an unique ID
				mShapes[i].ID = shapeUID;
				mShapes[i].lastFrameSeen = ci::app::getElapsedFrames();
				//mShapes[i].moving = true;
				// add this new shape to tracked shapes
				mTrackedShapes.push_back(mShapes[i]);
				shapeUID++;
			}
		}

		// if we didn't find a match for x frames, delete the tracked shape
		for (vector<Shape>::iterator it = mTrackedShapes.begin(); it != mTrackedShapes.end(); ) {
			if (ci::app::getElapsedFrames() - it->lastFrameSeen > 20) {
				// remove the tracked shape
				it = mTrackedShapes.erase(it);
			}
			else {
				++it;
			}
		}

		cv::Mat gray8Bit;
		withoutBlack.convertTo(gray8Bit, CV_8UC3, 0.1 / 1.0);

		mSurfaceDepth = Surface8u(fromOcv(mInput));
		mSurfaceBlur = Surface8u(fromOcv(withoutBlack));
		mSurfaceSubtract = Surface8u(fromOcv(eightBit));
	}
}

void ThesisTest1App::draw()
{
	// clear out the window with black
	gl::clear(Color(1, 1, 1));

	if (mSurfaceDepth.getWidth() > 0) {
		if (mTextureDepth) {
			mTextureDepth->update(Channel32f(mSurfaceDepth));
		}
		else {
			mTextureDepth = gl::Texture::create(Channel32f(mSurfaceDepth));
		}
		gl::color(Color::white());
		gl::draw(mTextureDepth, mTextureDepth->getBounds());
	}
	gl::pushMatrices();
	gl::translate(vec2(320, 0));

	if (mSurfaceBlur.getWidth() > 0) {
		if (mTextureDepth) {
			mTextureDepth->update(Channel32f(mSurfaceBlur));
		}
		else {
			mTextureDepth = gl::Texture::create(Channel32f(mSurfaceSubtract));
		}
		gl::draw(mTextureDepth, mTextureDepth->getBounds());
	}
	gl::translate(vec2(0, 240));

	if (mSurfaceSubtract.getWidth() > 0) {
		if (mTextureDepth) {
			mTextureDepth->update(Channel32f(mSurfaceSubtract));
		}
		else {
			mTextureDepth = gl::Texture::create(Channel32f(mSurfaceSubtract));
		}
		gl::draw(mTextureDepth, mTextureDepth->getBounds());
	}
	gl::translate(vec2(-320, 0));
	
	// draw shapes
	for (ContourVector::iterator iter = mContours.begin(); iter != mContours.end(); ++iter) {
		gl::begin(GL_LINE_LOOP);
		for (vector<cv::Point>::iterator pt = iter->begin(); pt != iter->end(); ++pt) {
			gl::color(Color(1.0f, 0.0f, 0.0f));
			gl::vertex(fromOcv(*pt));
		}
		gl::end();
	}
	gl::translate(vec2(0, 240));
	for (int i = 0; i < mTrackedShapes.size(); i++) {
		gl::begin(GL_POINTS);
		for (int j = 0; j < mTrackedShapes[i].hull.size(); j++) {
			//if (mTrackedShapes[i].moving) {
				gl::color(Color(1.0f, 0.0f, 0.0f));
				//vec2 v = fromOcv(mTrackedShapes[i].hull[j]);
				gl::vertex(fromOcv(mTrackedShapes[i].hull[j]));
			//}
		}
		gl::end();
	}
	gl::popMatrices();
	mParams->draw();
}

cv::Mat ThesisTest1App::removeBlack(cv::Mat input, short nearLimit, short farLimit)
{
	for (int y = 0; y < input.rows; y++) {
		for (int x = 0; x < input.cols; x++) {
			// if a shape is too close or too far away, set the depth to a fixed number
			if (input.at<short>(y, x) < nearLimit || input.at<short>(y, x) > farLimit) {
				input.at<short>(y, x) = farLimit;
			}
		}
	}
	return input;
}

vector< Shape > ThesisTest1App::getEvaluationSet(ContourVector rawContours, int minimalArea, int maxArea)
{
	vector< Shape > vec;
	for (vector<cv::Point> &c : rawContours) {
		// create a matrix for the contour
		cv::Mat matrix = cv::Mat(c);

		// extract data from contour
		cv::Scalar center = mean(matrix);
		double area = cv::contourArea(matrix);

		// reject it if too small
		if (area < minimalArea) {
			continue;
		}

		// reject it if too big
		if (area > maxArea) {
			continue;
		}

		// store data
		Shape shape;
		shape.area = area;
		shape.centroid = cv::Point(center.val[0], center.val[1]);

		// get depth value from center point
		//float centerDepth = (float)mInput.at<short>(shape.centroid.y, shape.centroid.x);
		// map 10 4000 to 0 1
		//shape.depth = lmap(centerDepth, (float)mNearLimit, (float)mFarLimit, 0.0f, 1.0f);

		// store points around shape
		shape.hull = c;
		shape.matchFound = false;
		vec.push_back(shape);
	}
	return vec;
}

Shape* ThesisTest1App::findNearestMatch(Shape trackedShape, vector< Shape > &shapes, float maximumDistance)
{
	Shape* closestShape = NULL;
	float nearestDist = 1e5;
	if (shapes.empty()) {
		return NULL;
	}

	// finalDist keeps track of the distance between the trackedShape and the chosen candidate
	//float finalDist;

	for (Shape &candidate : shapes) {
		// find dist between the center of the contour and the shape
		cv::Point distPoint = trackedShape.centroid - candidate.centroid;
		float dist = cv::sqrt(distPoint.x * distPoint.x + distPoint.y * distPoint.y);
		if (dist > maximumDistance) {
			continue;
		}
		if (candidate.matchFound) {
			continue;
		}
		if (dist < nearestDist) {
			nearestDist = dist;
			closestShape = &candidate;
			//finalDist = dist;
		}
	}

	// if a candidate was matched to the tracked shape
	//if (closestShape) {
	//	// if the shape isn't moving
	//	if (finalDist < 1.5) {
	//		// 'dilute' motion
	//		closestShape->motion = trackedShape.motion * .995;
	//		// if diluted motion is under the threshold or it was already not moving, the object is not moving
	//		if (closestShape->motion < 1.5 || trackedShape.moving == false) {
	//			closestShape->moving = false;
	//			closestShape->motion = 0;
	//		}
	//	}
	//	else if (finalDist > 19 || trackedShape.moving == true) {
	//		closestShape->moving = true;
	//		closestShape->motion = finalDist;
	//	}
	//}
	return closestShape;
}

CINDER_APP(ThesisTest1App, RendererGl)

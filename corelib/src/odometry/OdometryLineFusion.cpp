#include "rtabmap/core/OdometryInfo.h"
#include "rtabmap/core/Memory.h"
#include "rtabmap/core/Signature.h"
#include "rtabmap/core/RegistrationVis.h"
#include "rtabmap/core/util3d.h"
#include "rtabmap/core/util3d_transforms.h"
#include "rtabmap/core/util3d_registration.h"
#include "rtabmap/core/util3d_motion_estimation.h"
#include "rtabmap/core/util3d_filtering.h"
#include "rtabmap/core/util3d_surface.h"
#include "rtabmap/core/Optimizer.h"
#include "rtabmap/core/VWDictionary.h"
#include "rtabmap/core/Graph.h"
#include "rtabmap/utilite/ULogger.h"
#include "rtabmap/utilite/UTimer.h"
#include "rtabmap/utilite/UMath.h"
#include "rtabmap/utilite/UConversion.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <rtabmap/core/odometry/OdometryLineFusion.h>
#include <pcl/common/io.h>
#include <rtabmap/core/odometry/LineSegment.h>
#include <rtabmap/core/odometry/EDLines.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <QtMath>

#if _MSC_VER
	#define ISFINITE(value) _finite(value)
#else
	#define ISFINITE(value) std::isfinite(value)
#endif

namespace rtabmap {

rtabmap::OdometryLineFusion::OdometryLineFusion(const rtabmap::ParametersMap& parameters)
    : Odometry(parameters)
    , m_resolution(0.005f)
    , m_angleThreshold(M_PI_2)
    , m_init(false)
{
    UDEBUG("");


}

rtabmap::OdometryLineFusion::~OdometryLineFusion()
{
}

Odometry::Type rtabmap::OdometryLineFusion::getType()
{
    return Odometry::Type();
}

Transform rtabmap::OdometryLineFusion::computeTransform(SensorData& data, const Transform& guess, OdometryInfo* info)
{
    Transform output = Transform::getIdentity();
    if (!init(data))
        return output;

    m_srcFrame = m_lineExtractor->compute(data);
    m_lineExtractor->generateVoxelsDescriptors(data, m_srcFrame.lines(), 0.05f, 5, 4, 8, data.imageRaw().cols, data.imageRaw().rows, data.cameraModels()[0].cx(), data.cameraModels()[0].cy(), data.cameraModels()[0].fx(), data.cameraModels()[0].fy());
    m_srcFrame.setPrevIndex(m_dstFrame.index());
    Eigen::Matrix3f rot(Eigen::Matrix3f::Identity());
    Eigen::Vector3f trans(Eigen::Vector3f::Zero());
    Eigen::Matrix4f initPose(Eigen::Matrix4f::Identity());
    initPose.topLeftCorner(3, 3) = rot;
    initPose.topRightCorner(3, 1) = trans;

    m_srcFrame.setPose(initPose);
    pcl::KdTreeFLANN<LineSegment>::Ptr tree(new pcl::KdTreeFLANN<LineSegment>());
    tree->setInputCloud(m_dstFrame.lines());

    float error = 0;
    QMap<int, int> pairs;
    QMap<int, float> weights;
    Eigen::Matrix4f poseDelta = m_lineMatcher->step(m_srcFrame.lines(), m_dstFrame.lines(), tree, initPose, error, pairs, weights);
    if (error >= 1)
    {
        return output;
    }

    m_pose = poseDelta * m_pose;
    m_srcFrame.setPose(m_pose);
    FLFrame prevFrame = m_dstFrame;

    m_dstFrame = m_srcFrame;
    
    m_fromData = data;
    output.fromEigen4f(m_pose);
    return output;
}

bool OdometryLineFusion::init(SensorData& data)
{
    if (!m_init)
    {
        m_lineExtractor.reset(new FusedLineExtractor);
        m_lineMatcher.reset(new LineMatcher);

        m_dstFrame = m_lineExtractor->compute(data);
        m_lineExtractor->generateVoxelsDescriptors(data, m_dstFrame.lines(), 0.05f, 5, 4, 8, data.imageRaw().cols, data.imageRaw().rows, data.cameraModels()[0].cx(), data.cameraModels()[0].cy(), data.cameraModels()[0].fx(), data.cameraModels()[0].fy());

        m_pose = Eigen::Matrix4f::Identity();
        m_dstFrame.setPose(m_pose);
        m_dstFrame.setKeyFrame();

        //m_flFrames.insert(m_dstFrame.index(), m_dstFrame);

        m_init = true;
        return false;
    }
    return true;
}

}

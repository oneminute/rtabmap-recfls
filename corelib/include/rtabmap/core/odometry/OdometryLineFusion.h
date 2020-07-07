#ifndef ODOMETRYLINEFUSION_H_
#define ODOMETRYLINEFUSION_H_

#include <QObject>
#include <QMap>
#include <rtabmap/core/Odometry.h>
#include <rtabmap/core/Optimizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/pcl_base.h>
#include <rtabmap/core/Link.h>
#include <pcl/octree/octree.h>

#include "FusedLineExtractor.h"
#include "LineMatcher.h"
#include "LineSegment.h"

namespace rtabmap {

class Signature;
class Registration;
class Optimizer;

struct Voxel
{
public:
    Voxel()
        : key(0)
    {}

    union {
        quint64 key;
        struct {
            quint16 x;
            quint16 y;
            quint16 z;
            quint16 w;
        };
    };

    std::vector<int> indices;

    Eigen::Vector3f center;
    Eigen::Vector3f normal;
    pcl::octree::OctreeLeafNode<pcl::PointNormal>* node;
};

class RTABMAP_EXP OdometryLineFusion : public Odometry
{
public:
    OdometryLineFusion(const rtabmap::ParametersMap& parameters = rtabmap::ParametersMap());
    virtual ~OdometryLineFusion();


    // Inherited via Odometry
    virtual Odometry::Type getType() override;

    virtual Transform computeTransform(SensorData& data, const Transform& guess = Transform(), OdometryInfo* info = 0) override;

private:
    bool init(SensorData& data);

private:
    SensorData m_fromData;
    SensorData m_toData;

    QMap<quint64, Voxel> m_fromVoxels;
    QMap<quint64, Voxel> m_toVoxels;

    float m_resolution;
    float m_angleThreshold;

    bool m_init;
    FLFrame m_srcFrame;
    FLFrame m_dstFrame;

    QScopedPointer<FusedLineExtractor> m_lineExtractor;
    QScopedPointer<LineMatcher> m_lineMatcher;

    QMap<qint64, Eigen::Matrix4f> m_poses;
    //QList<Eigen::Matrix4f> m_relPoses;
    //QMap<RelInformation::KeyPair, RelInformation> m_relInfors;

    Eigen::Matrix4f m_pose;
};

}

#endif // ODOMETRYLINEFUSION_H_

#ifndef FUSEDLINEEXTRACTOR_H
#define FUSEDLINEEXTRACTOR_H

#include <QObject>
#include <QMap>

#include <pcl/common/common.h>
#include <pcl/features/eigen.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/octree/octree.h>
#include <pcl/search/kdtree.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <rtabmap/core/SensorData.h>

#include "CudaInternal.h"
#include "CudaInternal.h"
#include "LineSegment.h"
#include "FLFrame.h"
#include <cuda.h>

using namespace rtabmap;

class FusedLineExtractor : public QObject
{
    Q_OBJECT
public:
    explicit FusedLineExtractor(QObject* parent = nullptr);
    ~FusedLineExtractor();

    void init(SensorData& data);

    FLFrame compute(SensorData& data);

    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud() { return m_cloud; }
    pcl::PointCloud<pcl::PointXYZI>::Ptr allBoundary() { return m_allBoundary; }
    //pcl::PointCloud<pcl::Normal>::Ptr normals() { return m_normals; }
    QMap<int, pcl::PointCloud<pcl::PointXYZI>::Ptr>& groupPoints() { return m_groupPoints; }

    cv::Mat colorLinesMat() const { return m_colorLinesMat; }
    cv::Mat linesMat() const { return m_linesMat; }
    void generateDescriptors(pcl::PointCloud<LineSegment>::Ptr lines, float radius, int segments, int angleSegments, float width, float height, float cx, float cy, float fx, float fy);
    //pcl::PointCloud<LineSegment>::Ptr linesCloud() { return m_linesCloud; }
    /*void generateLineDescriptor(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, const cv::Mat& pointsMat,
        const Eigen::Vector3f& point, const Eigen::Vector3f& dir, const LineSegment& line, LineDescriptor3& desc, int offset,
        float cx, float cy, float fx, float fy, float width, float height, float r, int m, int n);
    Eigen::Vector2f projTo2d(const Eigen::Vector3f& v);
    bool available2dPoint(const Eigen::Vector2f& v);*/

private:
    bool m_init;
    cv::Mat m_boundaryMat;
    cv::Mat m_pointsMat;
    cv::Mat m_colorLinesMat;
    cv::Mat m_linesMat;
    std::vector<float3> m_points;
    //pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_allBoundary;
    //pcl::PointCloud<pcl::Normal>::Ptr m_normals;
    QMap<int, pcl::PointCloud<pcl::PointXYZI>::Ptr> m_groupPoints;
    //pcl::PointCloud<LineSegment>::Ptr m_linesCloud;
    cuda::GpuFrame m_frameGpu;
};

#endif // FUSEDLINEEXTRACTOR_H

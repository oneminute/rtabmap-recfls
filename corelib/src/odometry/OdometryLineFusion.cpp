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
    init(data);
    //extractLines(data);

    m_srcFrame = m_lineExtractor->compute(data);
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
    //RelInformation rel;
    //rel.setKey(m_srcFrame.index(), m_dstFrame.index());
    //rel.setTransform(poseDelta);
    //rel.setError(error);
    //m_relInfors.insert(rel.key(), rel);
    //optimize(m_dstFrame);

    //frame.setFrameIndex(m_poses.size());
    m_pose = poseDelta * m_pose;
    m_srcFrame.setPose(m_pose);
    FLFrame prevFrame = m_dstFrame;

    m_dstFrame = m_srcFrame;
    
    //m_frames.append(frame);
    //m_poses.insert(frame.frameIndex(), m_pose);
    //m_flFrames.insert(m_dstFrame.index(), m_dstFrame);

    m_fromData = data;
    output.fromEigen4f(m_pose);
    return output;
}

void OdometryLineFusion::extractLines(SensorData& data)
{
    if (!data.isValid())
        return;

    cv::Mat grayImage;
    cv::cvtColor(data.imageRaw(), grayImage, cv::COLOR_RGB2GRAY);
    EDLines lineHandler = EDLines(grayImage, SOBEL_OPERATOR);
    int linesCount = lineHandler.getLinesNo();
    std::vector<LS> lines = lineHandler.getLines();

    std::cout << "linesCount:" << linesCount << std::endl;
    std::cout << "has laser:" << !data.laserScanRaw().isEmpty() << std::endl;

    if (!data.laserScanRaw().isEmpty())
    {
        LaserScan scan = data.laserScanRaw();
        pcl::PointCloud<pcl::PointNormal>::Ptr cloudNormals = util3d::laserScanToPointCloudNormal(scan, scan.localTransform());
        cloudNormals = util3d::removeNaNNormalsFromPointCloud(cloudNormals);
        std::cout << "cloud size:" << cloudNormals->size() << std::endl;

        /*pcl::PointCloud<pcl::PointNormal>::Ptr filtered(new pcl::PointCloud<pcl::PointNormal>);
        pcl::VoxelGrid<pcl::PointNormal> sor;
        sor.setInputCloud(cloudNormals);
        sor.setLeafSize(0.005f, 0.005f, 0.005f);
        sor.filter(*filtered);
        std::cout << "filtered cloud size:" << filtered->size() << std::endl;*/

        pcl::octree::OctreePointCloudSearch<pcl::PointNormal> octree(m_resolution);
        octree.setInputCloud(cloudNormals);
        octree.addPointsFromInputCloud();

        double minX, minY, minZ, maxX, maxY, maxZ;
        octree.getBoundingBox(minX, minY, minZ, maxX, maxY, maxZ);

        int leafCount = octree.getLeafCount();
        int branchCount = octree.getBranchCount();

        std::cout << "[BoundaryExtractor::computeVBRG] branchCount:" << branchCount << ", leafCount:" << leafCount << std::endl;
        std::cout << "[BoundaryExtractor::computeVBRG] bounding box:" << minX << minY << minZ << maxX << maxY << maxZ << std::endl;

        std::vector<pcl::octree::OctreeKey> beVoxels;
        pcl::octree::OctreePointCloud<pcl::PointXYZ>::LeafNodeIterator it(&octree);
        while (it != octree.leaf_end())
        {
            Voxel voxel;
            pcl::octree::OctreeLeafNode<pcl::PointNormal>* node = reinterpret_cast<pcl::octree::OctreeLeafNode<pcl::PointNormal>*>(it.getCurrentOctreeNode());
            pcl::octree::OctreeContainerPointIndices container = it.getLeafContainer();
            pcl::octree::OctreeKey key = it.getCurrentOctreeKey();
            Eigen::Vector3f keyPos(key.x, key.y, key.z);
            std::vector<int> indices = container.getPointIndicesVector();
            Eigen::Vector3f center(Eigen::Vector3f::Zero());
            Eigen::Vector3f avgNm(Eigen::Vector3f::Zero());
            for (int i = 0; i < indices.size(); i++)
            {
                pcl::PointNormal pclPoint = cloudNormals->points[indices[i]];
                Eigen::Vector3f point = pclPoint.getVector3fMap();
                Eigen::Vector3f normal = pclPoint.getNormalVector3fMap();
                /*if (!normal.isZero())
                {
                    std::cout << key.x << ", " << key.y << ", " << key.z << ", points: " << indices.size() << std::endl;
                }*/
                center += point;
                avgNm += normal;
            }
            center /= indices.size();
            avgNm /= indices.size();
            avgNm.normalize();

            //quint16 x = qFloor((center.x() - minX) / m_resolution);
            //quint16 y = qFloor((center.y() - minY) / m_resolution);
            //quint16 z = qFloor((center.z() - minZ) / m_resolution);

            //voxel.x = x;
            //voxel.y = y;
            //voxel.z = z;

            voxel.indices = indices;
            voxel.center = center;
            voxel.normal = avgNm;
            //if (key.x == 841 && key.y == 1770)
            //{
            //    //std::cout << x << ", " << y << ", " << z << " -- " << key.x << ", " << key.y << ", " << key.z << " -- " << voxel.key << std::endl;
            //    std::cout << key.x << ", " << key.y << ", " << key.z << ", points: " << indices.size() << std::endl;
            //    for (int i = -1; i <= 1; i++)
            //    {
            //        for (int j = -1; j <= 1; j++)
            //        {
            //            for (int k = -1; k <= 1; k++)
            //            {
            //                pcl::octree::OctreeKey nKey(key.x + i, key.y + j, key.z + k);
            //                bool exist = octree.existLeaf(nKey.x, nKey.y, nKey.z);
            //                std::cout << "    " << nKey.x << ", " << nKey.y << ", " << nKey.z << " exist: " << exist << std::endl;
            //            }
            //        }
            //    }
            //}

            bool isBe = false;
            bool isFold = false;

            // check be
            {
                // 获取当前体素的法线垂切面
                Eigen::Vector3f u = avgNm.unitOrthogonal();
                Eigen::Vector3f v = avgNm.cross(u);

                // 获取分布在垂切面上的近邻体素集合
                Eigen::Isometry3f t;
                Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitY(), avgNm);
                std::vector<float> angles(5 * 5 * 3);
                int count = 0;
                /*if (key.x == 841 && key.y == 1770)
                {
                    
                }*/
                //for (int i = -2; i <= 2; i++)
                //{
                //    for (int j = -1; j <= 1; j++)
                //    {
                //        for (int k = -2; k <= 2; k++)
                //        {
                //            Eigen::Vector3f localPos(i, j, k);
                //            Eigen::Vector3f rotLocalPos = q * localPos;
                //            //std::cout << "    " << localPos.transpose() << ": " << rotLocalPos.transpose() << std::endl;
                //            pcl::octree::OctreeKey nKey(qRound(rotLocalPos.x()) + key.x, qRound(rotLocalPos.y()) + key.y, qRound(rotLocalPos.z()) + key.z);
                //            bool exist = octree.existLeaf(nKey.x, nKey.y, nKey.z);
                //            if (exist)
                //            {
                //                Eigen::Vector3f delta = rotLocalPos - keyPos;
                //                delta.normalize();
                //                float angle = std::atan2f(u.dot(delta), v.dot(delta));
                //                angles[count++] = angle;
                //                //std::cout << "    " << localPos.transpose() << ": " << rotLocalPos.transpose() << ": " << delta.transpose() << ": " << nKey.x << ", " << nKey.y << ", " << nKey.z << " angle: " << angle << std::endl;
                //            }
                //        }
                //    }
                //}
                for (int i = -2; i <= 2; i++)
                {
                    for (int j = -2; j <= 2; j++)
                    {
                        for (int k = -2; k <= 2; k++)
                        {
                            Eigen::Vector3f localPos(i, j, k);
                            localPos += keyPos;
                            //std::cout << "    " << localPos.transpose() << ": " << rotLocalPos.transpose() << std::endl;
                            pcl::octree::OctreeKey nKey(qFloor(localPos.x()), qFloor(localPos.y()), qFloor(localPos.z()));
                            bool exist = octree.existLeaf(nKey.x, nKey.y, nKey.z);
                            if (exist)
                            {
                                Eigen::Vector3f delta = localPos;
                                delta.normalize();
                                float angle = std::atan2f(u.dot(delta), v.dot(delta));
                                angles[count++] = angle;
                                //std::cout << "    " << localPos.transpose() << ": " << rotLocalPos.transpose() << ": " << delta.transpose() << ": " << nKey.x << ", " << nKey.y << ", " << nKey.z << " angle: " << angle << std::endl;
                            }
                        }
                    }
                }

                if (count <= 10)
                {
                    isBe = false;
                }
                else
                {
                    //std::cout << "    " << count << std::endl;
                    angles.resize(count);
                    std::sort(angles.begin(), angles.end());

                    float max_dif = FLT_MIN, dif;
                    // Compute the maximal angle difference between two consecutive angles
                    for (std::size_t i = 0; i < angles.size() - 1; ++i)
                    {
                        dif = angles[i + 1] - angles[i];
                        if (max_dif < dif)
                            max_dif = dif;
                    }
                    // Get the angle difference between the last and the first
                    dif = 2 * static_cast<float> (M_PI) - angles[angles.size() - 1] + angles[0];
                    if (max_dif < dif)
                        max_dif = dif;

                    // Check results
                    isBe = max_dif > m_angleThreshold;
                }

                // 计算投影线的atan2值

                // 排序

                // 统计最大差值
            }

            if (isBe)
            {
                beVoxels.push_back(key);
            }

            it++;
        }

        std::cout << "be count:" << beVoxels.size() << std::endl;
    }
}

bool OdometryLineFusion::init(SensorData& data)
{
    if (!m_init)
    {
        m_lineExtractor.reset(new FusedLineExtractor);
        m_lineMatcher.reset(new LineMatcher);

        m_dstFrame = m_lineExtractor->compute(data);
        m_lineExtractor->generateDescriptors(m_dstFrame.lines(), 0.05, 4, 8, data.imageRaw().cols, data.imageRaw().rows, data.cameraModels()[0].cx(), data.cameraModels()[0].cy(), data.cameraModels()[0].fx(), data.cameraModels()[0].fy());

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

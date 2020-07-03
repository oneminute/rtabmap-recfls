#include <opencv2/cudafilters.hpp>
#include <pcl/common/pca.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <QtMath>

#include <rtabmap/core/odometry/FusedLineExtractor.h>
#include <rtabmap/core/odometry/Utils.h>
#include <rtabmap/core/odometry/EDLines.h>
#include "cuda.hpp"

FusedLineExtractor::FusedLineExtractor(QObject* parent)
    : QObject(parent),
    m_init(false)
{

}

FusedLineExtractor::~FusedLineExtractor()
{
}

void FusedLineExtractor::init(SensorData& data)
{
    if (!m_init)
    {
        m_points.resize(data.imageRaw().cols * data.imageRaw().rows);
        cuda::Parameters params;
        cv::Mat rgbMat = data.imageRaw();
        cv::Mat depthMat = data.depthRaw();
        params.colorWidth = rgbMat.cols;
        params.colorHeight = rgbMat.rows;
        params.depthWidth = depthMat.cols;
        params.depthHeight = depthMat.rows;

        CameraModel model = data.cameraModels()[0];

        params.cx = model.cx();
        params.cy = model.cy();
        params.fx = model.fx();
        params.fy = model.fy();
        params.minDepth = 0.4f;
        params.maxDepth = 100.f;
        params.borderLeft = 26;
        params.borderRight = 8;
        params.borderTop = 4;
        params.borderBottom = 4;
        params.depthShift = 1000;
        params.normalKernalRadius = 20;
        params.normalKnnRadius = 0.1;
        params.boundaryEstimationRadius = 5;
        params.boundaryGaussianSigma = 4;
        params.boundaryGaussianRadius = 20;
        params.boundaryEstimationDistance = 0.01f;
        params.boundaryAngleThreshold = 45;
        params.classifyRadius = 20;
        params.classifyDistance = 0.2f;
        params.peakClusterTolerance = 5;
        params.minClusterPeaks = 2;
        params.maxClusterPeaks = 3;
        params.cornerHistSigma = 0.25f;

        m_frameGpu.parameters = params;
        m_frameGpu.allocate();

        m_init = true;
    }
}

FLFrame FusedLineExtractor::compute(SensorData& data)
{
    init(data);

    FLFrame flFrame;
    flFrame.setIndex(data.id());
    flFrame.setTimestamp(data.stamp());


    // 抽取edline直线
    cv::Mat grayImage;
    cv::cvtColor(data.imageRaw(), grayImage, cv::COLOR_RGB2GRAY);
    EDLines lineHandler = EDLines(grayImage, SOBEL_OPERATOR);
    m_linesMat = lineHandler.getLineImage();
    //cv::Mat edlinesMat = lineHandler.drawOnImage();
    m_colorLinesMat = cv::Mat(data.imageRaw().rows, data.imageRaw().cols, CV_8UC3, cv::Scalar(0, 0, 0));

    //cv::imshow("ed lines", edlinesMat);
    //cv::imshow("lines", linesMat);
    int linesCount = lineHandler.getLinesNo();
    std::vector<LS> lines = lineHandler.getLines();

    // 抽取出的直线集合放在这儿
     m_groupPoints.clear();
    for (int i = 0; i < linesCount; i++)
    {
        m_groupPoints.insert(i, pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>));
    }

    m_frameGpu.upload(data.depthRaw());

    // 用cuda抽取be点和折线点
    //TICK("extracting_boundaries");
    cuda::generatePointCloud(m_frameGpu);
    //TOCK("extracting_boundaries");

    //TICK("boundaries_downloading");
    m_frameGpu.boundaryMat.download(m_boundaryMat);
    m_frameGpu.pointsMat.download(m_pointsMat);
    //std::vector<float3> points;
    m_frameGpu.pointCloud.download(m_points);
    //std::vector<float3> normals;
    //m_frameGpu.pointCloudNormals.download(normals);
    std::vector<uchar> boundaries;
    m_frameGpu.boundaries.download(boundaries);

    // 开始2d和3d的比对。
    //m_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    m_allBoundary.reset(new pcl::PointCloud<pcl::PointXYZI>);
    //m_normals.reset(new pcl::PointCloud<pcl::Normal>);
    int negativeNum = 0;
    for(int i = 0; i < data.imageRaw().rows; i++) 
    {
        for(int j = 0; j < data.imageRaw().cols; j++) 
        {
            cv::Point coord(j, i);
            int index = i * data.imageRaw().cols + j;
            float3 value = m_points[index];
            uchar pointType = boundaries[index];
            ushort lineNo = m_linesMat.ptr<ushort>(i)[j];
            //pcl::PointXYZ pt;
            pcl::PointXYZI ptI;
            pcl::Normal normal;

            //pt.x = value.x;
            //pt.y = value.y;
            //pt.z = value.z;
            ptI.x = value.x;
            ptI.y = value.y;
            ptI.z = value.z;
            ptI.intensity = lineNo;

            //normal.normal_x = normals[index].x;
            //normal.normal_y = normals[index].y;
            //normal.normal_z = normals[index].z;

            int ptIndex = m_pointsMat.at<int>(coord);
            if (ptIndex < 0)
            {
                negativeNum++;
            }
            else
            {
                //m_cloud->push_back(pt);
                //m_normals->push_back(normal);
                ptIndex -= negativeNum;
                m_pointsMat.at<int>(coord) = ptIndex;
            }
            
            //std::cout << j << ", " << i << ": " << lineNo << std::endl;

            if (pointType > 0 && lineNo != 65535)
            {
                //Eigen::Vector2f pt2d(j, i);
                //LS line = lines[lineNo];
                //cv::Point cvLineDir = line.end - line.start;
                //Eigen::Vector2f lineDir(cvLineDir.x, cvLineDir.y);
                //lineDir.normalize();
                //Eigen::Vector2f vLineDir(lineDir.x(), -lineDir.y());
                //vLineDir.normalize();
                //Eigen::Vector3f avg(Eigen::Vector3f::Zero());
                //int count = 0;
                //for (int ni = -2; ni <= 2; ni++)
                //{
                //    for (int nj = -2; nj <= 2; nj++)
                //    {
                //        Eigen::Vector2f pt2dN = pt2d + vLineDir * ni + lineDir * nj;
                //        Eigen::Vector2i pt2dNI = pt2dN.cast<int>();
                //        if (pt2dNI.x() < 0 || pt2dNI.x() >= frame.getDepthWidth() || pt2dNI.y() < 0 || pt2dNI.y() >= frame.getDepthHeight())
                //            continue;
                //        int ptIndexN = pt2dNI.y() * frame.getDepthHeight() + pt2dNI.x();
                //        float3 valueN = points[ptIndexN];
                //        uchar pointTypeN = boundaries[ptIndexN];
                //        if (pointTypeN <= 0)
                //            continue;
                //        avg += toVector3f(valueN);
                //        count++;
                //    }
                //}
                //avg /= count;
                ////std::cout << j << ", " << i << ": " << lineNo << ", count = " << count << ", avg = " << avg.transpose() << std::endl;

                //if (ptI.z <= avg.z())
                //{
                    m_allBoundary->points.push_back(ptI);
                    m_groupPoints[lineNo]->points.push_back(ptI);
                //}
            }
        }
    }
    /*m_cloud->width = m_cloud->points.size();
    m_cloud->height = 1;
    m_cloud->is_dense = true;
    m_normals->width = m_normals->points.size();
    m_normals->height = 1;
    m_normals->is_dense = true;*/
    m_allBoundary->width = m_allBoundary->points.size();
    m_allBoundary->height = 1;
    m_allBoundary->is_dense = true;

    //m_linesCloud.reset(new pcl::PointCloud<LineSegment>);

    for (int i = 0; i < linesCount; i++)
    {
        if (m_groupPoints[i]->size() < 10)
            continue;

        Eigen::Vector3f gCenter(Eigen::Vector3f::Zero());
        for (int j = 0; j < m_groupPoints[i]->points.size(); j++)
        {
            Eigen::Vector3f np = m_groupPoints[i]->points[j].getArray3fMap();
            gCenter += np;
        }
        gCenter /= m_groupPoints[i]->points.size();

        // 因为同一直线编号的点集中可能既有真点也有veil点，所以先做区域分割。
        pcl::IndicesClusters clusters;
        pcl::EuclideanClusterExtraction<pcl::PointXYZI> ece;
        ece.setClusterTolerance(0.05f);
        ece.setInputCloud(m_groupPoints[i]);
        ece.setMinClusterSize(1);
        ece.setMaxClusterSize(m_groupPoints[i]->points.size());
        ece.extract(clusters);

        //std::cout << i << ": " << "count = " << m_groupPoints[i]->points.size() << std::endl;

        int maxSize = 0;
        int maxIndex = 0;
        // 分割后，找出点数最多的子区域作为初始内点集合。即cloud。
        for (int j = 0; j < clusters.size(); j++)
        {
            Eigen::Vector3f clusterCenter(Eigen::Vector3f::Zero());
            for (int n = 0; n < clusters[j].indices.size(); n++)
            {
                Eigen::Vector3f np = m_groupPoints[i]->points[clusters[j].indices[n]].getArray3fMap();
                clusterCenter += np;
            }
            clusterCenter /= clusters[j].indices.size();
            bool valid = true;
            if (clusterCenter.z() > gCenter.z())
            {
                float dist = clusterCenter.z() - gCenter.z();
                if (dist > 0.03f)
                {
                    valid = false;
                }
            }
            //std::cout << "  sub " << j << ", count = " << clusters[j].indices.size() << ", farer = " << (clusterCenter.z() > gCenter.z()) << ", z dist = " << (clusterCenter.z() - gCenter.z())
                //<< ", valid = " << valid << std::endl;
            if (valid && clusters[j].indices.size() > maxSize)
            {
                maxSize = clusters[j].indices.size();
                maxIndex = j;
            }
        }
        if (maxSize < 3)
            continue;

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::copyPointCloud(*m_groupPoints[i], clusters[maxIndex].indices, *cloud);
        //pcl::copyPointCloud(*m_groupPoints[i], *cloud);

        // 计算这个初始内点集合的主方向和中点。
        pcl::PCA<pcl::PointXYZI> pca;
        pca.setInputCloud(cloud);
        Eigen::Vector3f eigenValues = pca.getEigenValues();
        float sqrt1 = sqrt(eigenValues[0]);
        float sqrt2 = sqrt(eigenValues[1]);
        float sqrt3 = sqrt(eigenValues[2]);
        float a1 = (sqrt1 - sqrt2) / sqrt1;
        float a2 = (sqrt2 - sqrt3) / sqrt1;
        float a3 = sqrt3 / sqrt1;

        //std::cout << "  " << m_groupPoints[i]->size() << ", cluster: " << clusters.size() << ", a1 = " << a1 << ", a2 = " << a2 << ", a3 = " << a3 << std::endl;
        //std::cout << "  init inliers size: " << cloud->size() << std::endl;

        // 主方向
        Eigen::Vector3f dir = pca.getEigenVectors().col(0).normalized();
        // 中点
        Eigen::Vector3f center = pca.getMean().head(3);

        // 然后，遍历剩余的子区域点集，查看每一个点到这条直线的距离是否在阈值以内，在就加到内点集中，不在就抛弃。
        for (int j = 0; j < clusters.size(); j++)
        {
            if (j == maxIndex)
                continue;

            for (int n = 0; n < clusters[j].indices.size(); n++)
            {
                int nIndex = clusters[j].indices[n];
                pcl::PointXYZI pclPt = m_groupPoints[i]->points[nIndex];
                Eigen::Vector3f pt = pclPt.getArray3fMap();
                float dist = (pt - center).cross(dir).norm();
                // 暂时硬编码的阈值。
                if (dist <= 0.05f)
                {
                    cloud->points.push_back(pclPt);
                }
            }
        }
        if (cloud->size() < 10)
            continue;

        // 最后再计算一遍内点集的主方向与中点。
        //std::cout << "    final: " << cloud->size() << ", max size: " << maxSize << ", max index: " << maxIndex << std::endl;
        //std::cout << "    final: " << cloud->size() << std::endl;
        pcl::PCA<pcl::PointXYZI> pcaFinal;
        pcaFinal.setInputCloud(cloud);
        eigenValues = pcaFinal.getEigenValues();
        dir = pcaFinal.getEigenVectors().col(0).normalized();
        center = pcaFinal.getMean().head(3);
        //Eigen::Vector3f eigenValues = pcaFinal.getEigenValues();
        //Eigen::Vector3f dir = pcaFinal.getEigenVectors().col(0).normalized();
        //Eigen::Vector3f center = pcaFinal.getMean().head(3);

        // 确定端点。
        Eigen::Vector3f start(0, 0, 0);
        Eigen::Vector3f end(0, 0, 0);
        for (int j = 0; j < cloud->size(); j++)
        {
            pcl::PointXYZI& ptBoundary = cloud->points[j];
            Eigen::Vector3f boundaryPoint = ptBoundary.getVector3fMap();
            Eigen::Vector3f projPoint = closedPointOnLine(boundaryPoint, dir, center);

            if (start.isZero())
            {
                // 如果第一次循环，让当前点作为起点
                start = projPoint;
            }
            else
            {
                // 将当前点与当前计算出的临时起点连在一起，查看其与当前聚集主方向的一致性，若一致则当前点为新的起点。
                if ((start - projPoint).dot(dir) > 0)
                {
                    start = projPoint;
                }
            }

            if (end.isZero())
            {
                // 如果第一次循环，让当前点作为终点
                end = projPoint;
            }
            else
            {
                // 将当前点与当前计算出的临时终点连在一起，查看其与当前聚集主方向的一致性，若一致则当前点为新的终点。
                if ((projPoint - end).dot(dir) > 0)
                {
                    end = projPoint;
                }
            }
        }

        LS line2d = lines[i];
        LineSegment line;
        line.setStart(start);
        line.setEnd(end);
        line.setStart2d(line2d.start);
        line.setEnd2d(line2d.end);
        line.calculateColorAvg(data.imageRaw());
        line.drawColorLine(m_colorLinesMat);
        //line.reproject();
        //std::cout << line.shortDescriptorSize() << std::endl;
        line.setIndex(flFrame.lines()->points.size());
        if (line.length() > 0.1f)
        {
            //m_lines.insert(i, line);
            flFrame.lines()->points.push_back(line);
        }
        flFrame.lines()->width = flFrame.lines()->points.size();
        flFrame.lines()->height = 1;
        flFrame.lines()->is_dense = true;
    }
    
    qDebug() << "all boundary points:" << m_allBoundary->size();
    //TOCK("boundaries_downloading");

    return flFrame;
}

void FusedLineExtractor::generateDescriptors(pcl::PointCloud<LineSegment>::Ptr lines, float radius, int segments, int angleSegments, float width, float height, float cx, float cy, float fx, float fy)
{
    cv::Mat board(height, width, CV_8UC3, cv::Scalar(0));
    for (int i = 0; i < lines->points.size(); i++)
    {
        LineSegment ls = lines->points[i];
        cv::Point2f cvDir2d = ls.end2d() - ls.start2d();
        Eigen::Vector3f vDir(cvDir2d.y, -cvDir2d.x, 0);
        vDir.normalize();
        Eigen::Vector3f pa = ls.start() - vDir * radius;
        Eigen::Vector3f pb = ls.start() + vDir * radius;
        Eigen::Vector3f pc = ls.end() + vDir * radius;
        Eigen::Vector3f pd = ls.end() - vDir * radius;
        std::vector<cv::Point2f> contours(4);
        contours[0] = cv::Point2f(pa.x() * fx / pa.z() + cx, pa.y() * fy / pa.z() + cy);
        contours[1] = cv::Point2f(pb.x() * fx / pb.z() + cx, pb.y() * fy / pb.z() + cy);
        contours[2] = cv::Point2f(pc.x() * fx / pc.z() + cx, pc.y() * fy / pc.z() + cy);
        contours[3] = cv::Point2f(pd.x() * fx / pd.z() + cx, pd.y() * fy / pd.z() + cy);
        //contours[4] = contours[0];
        cv::Rect rect = cv::boundingRect(contours);

        int histSize = segments * angleSegments;
        std::vector<Eigen::Vector3f> neighbours;
        std::vector<int> hist(histSize);
        //std::vector<int> tmpHist(histSize);
        for (int n = 0; n < histSize; n++)
        {
            hist[n] = 0;
            //tmpHist[n] = 0;
        }

        std::vector<int> angles(angleSegments);
        for (int n = 0; n < angleSegments; n++)
        {
            angles[n] = 0;
        }
        
        int maxAngleIndex = 0;
        int maxAngleCount = 0;
        for (int ix = 0; ix < rect.width; ix++)
        {
            for (int iy = 0; iy < rect.height; iy++)
            {
                cv::Point pixel(rect.x + ix, rect.y + iy);
                if (pixel.x >= 0 && pixel.x < width && pixel.y >= 0 && pixel.y < height)
                {
                    if (cv::pointPolygonTest(contours, pixel, false) >= 0)
                    {
                        board.at<cv::Vec3b>(pixel) = cv::Vec3b(0, 255, 255);
                        float3 point3 = m_points[pixel.y * width + pixel.x];
                        Eigen::Vector3f point(point3.x, point3.y, point3.z);
                        Eigen::Vector3f edge = point - ls.start();
                        Eigen::Vector3f dir = ls.direction().normalized();
                        float dist = edge.cross(dir).norm();
                        if (dist <= radius)
                        {
                            neighbours.push_back(point);
                            Eigen::Vector3f vert = edge.cross(dir).cross(dir).normalized();
                            float radians = qAtan2(vDir.dot(vert), dir.dot(vert)) + M_PI;
                            int angleSeg = radians * angleSegments / (M_PI * 2);
                            int distSeg = qFloor(dist * segments / radius);
                            angles[angleSeg]++;
                            hist[angleSeg * segments + distSeg]++;
                            if (angles[angleSeg] > maxAngleCount)
                            {
                                maxAngleIndex = angleSeg;
                                maxAngleCount = angles[angleSeg];
                            }
                        }
                    }
                }
            }
        }

        if (neighbours.size() >= 3)
        {
            for (int n = 0; n < histSize; n++)
            {
                int newIndex = (n + histSize - maxAngleIndex * segments) % histSize;
                int tmp = hist[newIndex];
                hist[newIndex] = hist[n];
                hist[n] = tmp;
            }
        }

        for (int n = 0; n < histSize; n++)
        {
            std::cout << std::setw(4) << hist[n] << " ";
        }
        std::cout << std::endl;
        //if (i == 0)
        {
            cv::line(board, ls.start2d(), ls.end2d(), cv::Scalar(255, 0, 0));
            std::vector<std::vector<cv::Point2f>> cc;
            cc.push_back(contours);
            cv::line(board, contours[0], contours[1], cv::Scalar(0, 0, 255));
            cv::line(board, contours[1], contours[2], cv::Scalar(0, 0, 255));
            cv::line(board, contours[2], contours[3], cv::Scalar(0, 0, 255));
            cv::line(board, contours[3], contours[0], cv::Scalar(0, 0, 255));
            cv::rectangle(board, rect, cv::Scalar(0, 255, 0));
        }
    }
    cv::imwrite("desc01.png", board);
}

//void FusedLineExtractor::generateLineDescriptor(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, const cv::Mat& pointsMat,
//    const Eigen::Vector3f& point, const Eigen::Vector3f& dir, const LineSegment& line, LineDescriptor3& desc, int offset,
//    float cx, float cy, float fx, float fy, float width, float height, float r, int m, int n)
//{
//    Eigen::Vector3f start = point - line.dir * r;
//    Eigen::Vector3f end = point + line.dir * r;
//    Eigen::Vector2f start2d = projTo2d(start);
//    Eigen::Vector2f end2d = projTo2d(end);
//    Eigen::Vector2f dir2d = (end2d - start2d).normalized();
//    Eigen::Vector2f vert2d(-dir2d.y(), dir2d.x());
//    Eigen::Vector3f localZ = line.dir.cross(dir).normalized();
//
//    float length2d = (start2d - end2d).norm();
//    qDebug()
//        << "start: [" << start.x() << start.y() << start.z()
//        << "], end: [" << end.x() << end.y() << end.z()
//        << "], start2d: [" << start2d.x() << start2d.y()
//        << "], end2d: [" << end2d.x() << end2d.y()
//        << "], length: " << length2d;
//    for (int o = -n; o <= n; o++)
//    {
//        //qDebug() << o;
//        for (int j = -m; j <= m; j++)
//        {
//            int positiveNum = 0;
//            int negativeNum = 0;
//
//            Eigen::Vector2f pt = start2d + (vert2d * (m * 2 + 1) * o);
//            //qDebug() << pt.x() << pt.y();
//            for (int i = 0; i <= floor(length2d); i++)
//            {
//                Eigen::Vector2f pt2 = pt + vert2d * j + dir2d * i;
//                if (!available2dPoint(pt2))
//                    continue;
//
//                cv::Point2i pt2Pix(qFloor(pt2.x()), qFloor(pt2.y()));
//                int ptIndex = pointsMat.at<int>(pt2Pix);
//                if (ptIndex < 0)
//                    continue;
//                if (ptIndex >= cloud->points.size())
//                    continue;
//
//                Eigen::Vector3f pt3d = cloud->points[ptIndex].getVector3fMap();
//                Eigen::Vector3f nm3d = normals->points[ptIndex].getNormalVector3fMap();
//
//                Eigen::Vector3f diff = pt3d - point;
//                if (diff.dot(dir) >= 0)
//                {
//                    positiveNum++;
//                }
//                else
//                {
//                    negativeNum++;
//                }
//
//                Eigen::Vector3f projNm = nm3d - localZ * nm3d.dot(localZ);
//                projNm.normalize();
//                float angles = qAtan2(projNm.dot(line.dir), projNm.dot(dir)) + M_PI;
//                int subIndex = qFloor(angles * 4 / M_PI);
//                int dim = offset + (o + n) * 10 + subIndex;
//                desc.elems[dim]++;
//                if (dim < 0 || dim >= LineDescriptor3::elemsSize())
//                {
//                    qDebug() << dim;
//                }
//
//                cv::Vec3b& color = m_board.at<cv::Vec3b>(pt2Pix);
//                color[0] = (o + n) * (256 / (n * 2 + 1));
//                color[1] = color[0] + (j + m);
//                color[2] = i;
//            }
//            int dim = offset + (o + n) * 10 + 8;
//            if (dim < 0 || dim >= LineDescriptor3::elemsSize())
//            {
//                qDebug() << dim;
//            }
//            dim = offset + (o + n) * 10 + 9;
//            if (dim < 0 || dim >= LineDescriptor3::elemsSize())
//            {
//                qDebug() << dim;
//            }
//            desc.elems[offset + (o + m) * 10 + 8] = positiveNum;
//            desc.elems[offset + (o + m) * 10 + 9] = negativeNum;
//        }
//    }
//}
//
//Eigen::Vector2f FusedLineExtractor::projTo2d(const Eigen::Vector3f& v)
//{
//    Eigen::Vector2f v2d(v.x() * m_fx / v.z() + m_cx, v.y() * m_fy / v.z() + m_cy);
//    return v2d;
//}
//
//bool FusedLineExtractor::available2dPoint(const Eigen::Vector2f& v)
//{
//    if (v.x() >= 0 && v.x() < m_matWidth && v.y() >= 0 && v.y() < m_matHeight)
//        return true;
//    return false;
//}

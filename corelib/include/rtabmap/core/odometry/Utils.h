#ifndef UTILS_H
#define UTILS_H

#include <QObject>
#include <QDebug>
#include <QDataStream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

enum uCvQtDepthColorMap{
    uCvQtDepthWhiteToBlack,
    uCvQtDepthBlackToWhite,
    uCvQtDepthRedToBlue,
    uCvQtDepthBlueToRed
};

class Utils
{
public:
    Utils();

    static void registerTypes();
};

template<typename _Scalar, int _Rows, int _Cols>
QDebug qDebugMatrix(QDebug &out, const Eigen::Matrix<_Scalar, _Rows, _Cols>& m);

template<typename _Scalar, int _Rows, int _Cols>
QDataStream &streamInMatrix(QDataStream &in, Eigen::Matrix<_Scalar, _Rows, _Cols> &m);

QDebug operator<<(QDebug out, const Eigen::Matrix4f &m);

QDebug operator<<(QDebug out, const cv::Mat &m);

QDebug operator<<(QDebug out, const Eigen::Vector3f &v);

QDataStream &operator>>(QDataStream &in, Eigen::Matrix4f &m);

QImage cvMat2QImage(const cv::Mat & image, bool isBgr = true, uCvQtDepthColorMap colorMap = uCvQtDepthWhiteToBlack);

Eigen::Matrix4f matrix4fFrom(float x, float y, float z, float roll, float pitch, float yaw);

Eigen::Matrix4f matrix4fFrom(float x, float y, float theta);

Eigen::Affine3f affine3fFrom(const Eigen::Matrix4f &m);

Eigen::Quaternionf quaternionfFrom(const Eigen::Matrix4f &m);

Eigen::Matrix4f normalizeRotation(const Eigen::Matrix4f &m);

Eigen::Vector3f transformPoint(const Eigen::Vector3f &p, const Eigen::Matrix4f &m);

Eigen::Vector3f pointFrom(const Eigen::Matrix4f &m);

Eigen::Vector4f vector4fZeroFrom(const Eigen::Matrix4f &m);

Eigen::Matrix4f rotationFrom(const Eigen::Matrix4f &m);

Eigen::Matrix4f translationFrom(const Eigen::Matrix4f &m);

cv::Mat cvMatFrom(const Eigen::MatrixXf &m);

template<class T>
int sign(const T &v)
{
    if (v < 0)
        return -1;
    else
        return 1;
}

template<class T>
void findMinMax(const T *v, quint32 size, T &min, T &max, quint32 &minIndex, quint32 &maxIndex)
{
    min = std::numeric_limits<T>::max();
    max = std::numeric_limits<T>::min();
    minIndex = 0;
    maxIndex = 0;

    if (v == nullptr || size == 0)
    {
        return;
    }

    min = v[0];
    max = v[0];

    for (quint32 i = 0; i < size; i++)
    {
        if (qIsNaN(min) || (min > v[i] && !qIsNaN(v[i])))
        {
            min = v[i];
            minIndex = i;
        }

        if (qIsNaN(max) || (max < v[i] && !qIsNaN(v[i])))
        {
            max = v[i];
            maxIndex = i;
        }
    }
}

template<class T>
void findMinMax(const T *v, quint32 size, T &min, T &max)
{
    quint32 minIndex = 0;
    quint32 maxIndex = 0;
    findMinMax(v, size, min, max, minIndex, maxIndex);
}

Eigen::Vector3f closedPointOnLine(const Eigen::Vector3f &point, const Eigen::Vector3f &dir, const Eigen::Vector3f &meanPoint);

float oneAxisCoord(const Eigen::Vector3f& point, const Eigen::Vector3f& dir);

void calculateAlphaBeta(const Eigen::Vector3f& dir, float& alpha, float& beta);

float distanceBetweenLines(const Eigen::Vector3f& line1, const Eigen::Vector3f& point1, const Eigen::Vector3f& line2, const Eigen::Vector3f& point2);

Eigen::Vector3f transBetweenLines(const Eigen::Vector3f& line1, const Eigen::Vector3f& point1, const Eigen::Vector3f& line2, const Eigen::Vector3f& point2, float& distance);

#endif // UTILS_H

package leastsquares;

import java.util.ArrayList;
import java.util.HashMap;
import java.awt.* ;
import static java.lang.Math.*;

import org.apache.commons.math3.fitting.leastsquares.*;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.optim.SimplePointChecker;
import org.apache.commons.math3.optim.SimpleValueChecker;
import org.apache.commons.math3.util.Pair;

public class LocateCamera {
    ArrayList<DetectedTag> tags = new ArrayList<DetectedTag>();
    HashMap<Integer, Point> landmarks = new HashMap<Integer, Point>();

    double[] distanceBAP;
    double[] cosAngleBAP;
    double[] cosAngle1to3;
    PointPair[] pointPairs;
    double[] estimatedDTT;
    double lastRms = 0;

    public void addDetectedTag(DetectedTag dt) {
        tags.add(dt);
    }

    public void clearTags() {
        tags.clear();
    }

    public void addLandmark(int id, Point lm) {
        landmarks.put(id, lm);
    }
    
    public Result solve() {
        
        //Sort by increasing angle
        tags.sort(null);
        int nTags = tags.size();

        //Make and intial estimate of position based on the estimated distances to
        //the landmarks.  Use 1st and last (as viewed is order of increasing angle)
        //landmarks.
        DetectedTag tag1 = tags.get(0);
        DetectedTag tagN = tags.get(nTags-1);
        Point pointA = landmarks.get(tag1.id);
        Double distanceA = tag1.distance;
        Point pointB = landmarks.get(tagN.id);
        Double distanceB = tagN.distance;
        Double pointAAngle = tag1.angle;    //For use in final observer pose clac
        Double pointsAngle = tagN.angle - tag1.angle;
        Pose2D estimatedPose = observerPosition(pointA, pointB, distanceA, distanceB, pointsAngle);

        //Compute the distances between adjacent observed tags
        int nSize = (nTags - 1) + (nTags - 1)/2;
        distanceBAP = new double[nSize];
        cosAngleBAP = new double[nSize];
        pointPairs = new PointPair[nSize];
        for(int i = 0; i < (nTags - 1); ++i) {
            int id1 = tags.get(i).id;
            int id2 = tags.get(i + 1).id;
            Point point1 = landmarks.get(id1);
            Point point2 = landmarks.get(id2);
            pointPairs[i] = new PointPair(point1, point2);
            distanceBAP[i] = point1.distance(point2);
            cosAngleBAP[i] = cos(tags.get(i+1).angle - tags.get(i).angle);
        }

        //Each set of three adjacent observed tags forms a triangle.
        //Compute the length of the side formed by the 1st and third point
        int n1to3 = (nTags - 1) / 2;     
        for (int i = 0; i < n1to3; ++i) {
            int j = i + (nTags - 1);
            int id1 = tags.get(i * 2).id;
            int id2 = tags.get(i * 2 + 2).id;
            Point point1 = landmarks.get(id1);
            Point point2 = landmarks.get(id2);
            pointPairs[j] = new PointPair(point1, point2);
            distanceBAP[j] = point1.distance(point2);
            //cosAngleBAP[j] = cos(tags.get(i*2+2).angle - tags.get(i*2).angle);
            cosAngleBAP[j] = cos(tags.get(i*2+2).angle - tags.get(i*2).angle);
        }

        //Build the array of estimated distances from camera to tag
        estimatedDTT = new double[nTags];
        for (int i = 0; i < nTags; ++i) {
            estimatedDTT[i] = tags.get(i).distance;
        }

        double[] startXY = {estimatedPose.xy.x, estimatedPose.xy.y};
        LeastSquaresProblem problem = new LeastSquaresBuilder().
            start(startXY).
            model(triangulate).
            target(cosAngleBAP).
            lazyEvaluation(false).
            maxEvaluations(1000).
            maxIterations(1000).
            checker(new EvaluationRmsChecker(1e-12)).
            build();

        LeastSquaresOptimizer.Optimum optimum = new LevenbergMarquardtOptimizer().optimize(problem);
        double x1 = optimum.getCost();
        double x2 = optimum.getRMS();
        RealVector v1 = optimum.getResiduals();

        double xResult = optimum.getPoint().getEntry(0);
        double yResult = optimum.getPoint().getEntry(1);
        
        double obsAngle = atan2(pointA.y-yResult, pointA.x-xResult) - pointAAngle;
        Pose2D obsPose = new Pose2D(new Point(xResult, yResult), obsAngle);

        Result result = new Result(obsPose, optimum.getRMS(), optimum.getIterations());
        
        return result;
    }

    MultivariateJacobianFunction triangulate = new MultivariateJacobianFunction() {

        public Pair<RealVector, RealMatrix> value(final RealVector vDTC) {         
            int numYs = distanceBAP.length;
            int numXs = vDTC.getDimension();
        
            RealVector vecCosAngle = new ArrayRealVector(numYs);
            RealMatrix jacobian = new Array2DRowRealMatrix(numYs, numXs);
            
            //Compute the cosine of the angle opposite side C 
            for(int i = 0; i < distanceBAP.length; ++i) {
                double X = vDTC.getEntry(0);
                double Y = vDTC.getEntry(1);
                Point A = pointPairs[i].A;
                Point B = pointPairs[i].B;
                double Csq = distanceBAP[i] * distanceBAP[i];
                double[] xCalc = derivX(A, B, Csq, X, Y);
                double[] yCalc = derivY(A, B, Csq, X, Y);

                vecCosAngle.setEntry(i, xCalc[0] ) ; 
                jacobian.setEntry(i, 0, xCalc[1]);
                jacobian.setEntry(i, 1, yCalc[1]);
                
                //Following code for debug only
                double rms = 0;
                for (int j = 0; i < cosAngleBAP.length; ++i) {
                    double x1 = cosAngleBAP[i];
                    double x2 = vecCosAngle.getEntry(j);
                    rms += (x1-x2)*(x1-x2);
                }
                rms = sqrt(rms);
                double diff = rms-lastRms;
                lastRms = rms;
               
            } 

            return new Pair<RealVector, RealMatrix>(vecCosAngle, jacobian);
        }
    };

    double[] derivX(Point A, Point B, double Csq, double X, double Y) {
        /* A and B are points in 2D Cartesian space that form the base of a triangle.
           X and Y are the coordinates of the point C that is the apex of the triangle.
           Csq is the squared length of the base (passed in because it is constant).
           In the espression below, AC and BC are defined as the lengths
           of the two remaining sides.

           This code computes the cosine of the apex angle and the partial derivative of
           same with respect to X.

           cos(theta) = (AC^2 + BC^2 - Csq) / (2AC*BC)
         */
        
        //Distance C to A squared
        double F1 = (X-A.x)*(X-A.x) + (Y-A.y)*(Y-A.y);
        double dF1_dx = 2*X-2*A.x;

        //Distance C to B squared
        double F2 = (X-B.x)*(X-B.x) + (Y-B.y)*(Y-B.y);
        double dF2_dx = 2*X-2*B.x;

        //Product of squared distance
        double F3 = F1 * F2;
        double dF3_dF1 = F2;
        double dF3_dF2 = F1;
        double dF3_dx = dF3_dF1 * dF1_dx + dF3_dF2 * dF2_dx;

        //Denominator expression
        double F4 = 2*sqrt(F3);
        double dF4_dF3 = 1/sqrt(F3);
        double dF4_dx = dF4_dF3 * dF3_dx;

        //Inverse of denominator
        double F5 = 1 / F4;
        double dF5_dF4 = -1/(F4 * F4);
        double dF5_dx = dF5_dF4 * dF4_dx;

        //Numerator expression
        double F6 = F1 + F2 - Csq;
        double dF6_dx = dF1_dx + dF2_dx;
        
        //Final expression
        double F7 = F6 * F5;
        double dF7_dF6 = F5;
        double dF7_dF5 = F6;
        double dF7_dx = dF7_dF6 * dF6_dx + dF7_dF5 * dF5_dx;

        double[] result = new double[2];
        result[0] = F7;
        result[1] = dF7_dx;
        return result;
    }

    double[] derivY(Point A, Point B, double Csq, double X, double Y) {
        //Distance to A squared
        double F1 = (X-A.x)*(X-A.x) + (Y-A.y)*(Y-A.y);
        double dF1_dy = 2*Y-2*A.y;

        //Distance to B squared
        double F2 = (X-B.x)*(X-B.x) + (Y-B.y)*(Y-B.y);
        double dF2_dy = 2*Y-2*B.y;

        //Product of squared distance
        double F3 = F1 * F2;
        double dF3_dF1 = F2;
        double dF3_dF2 = F1;
        double dF3_dy = dF3_dF1 * dF1_dy + dF3_dF2 * dF2_dy;

        //Denominator expression
        double F4 = 2*sqrt(F3);
        double dF4_dF3 = 1/sqrt(F3);
        double dF4_dy = dF4_dF3 * dF3_dy;

        //Inverse of denominator
        double F5 = 1 / F4;
        double dF5_dF4 = -1/(F4 * F4);
        double dF5_dy = dF5_dF4 * dF4_dy;

        //Numerator expression
        double F6 = F1 + F2 - Csq;
        double dF6_dy = dF1_dy + dF2_dy;
        
        //Final expression
        double F7 = F6 * F5;
        double dF7_dF6 = F5;
        double dF7_dF5 = F6;
        double dF7_dy = dF7_dF6 * dF6_dy + dF7_dF5 * dF5_dy;

        double[] result = new double[2];
        result[0] = F7;
        result[1] = dF7_dy;
        return result;
    }
    
    //Computes the pose of the observer from a triangle whose base is the line between two
    //landmarks (base 1 and base 2), the length of the line from the observer to
    //base1 (a), and the lenght of the line from the observer to base2 (b).
    private Pose2D observerPosition(Point base1, Point base2, double a, double b, double obsAngle) {
        
        //Compute the angle of the base with respect to the world coordinate system
        //and the length of the base
        double theta1 = atan2(base2.y - base1.y, base2.x - base1.x);
        double c = base1.distance(base2);
        
        //Compute the angle opposite side b
        double theta2 = acos((a*a + c*c - b*b) / (2*a*c));
        double x = base1.x + a*cos(theta1+theta2);
        double y = base1.y + a*sin(theta1+theta2);
        double angle = theta1 + theta2 - obsAngle - PI;
        return new Pose2D(new Point(x, y), unwrapAngle(angle));
    }

    private double unwrapAngle(double angle) {
        while (angle > PI) angle -= PI;
        while (angle < (-PI)) angle += PI;
        return angle;
    }

    public static class DetectedTag implements Comparable<DetectedTag> {
        int id;
        public double angle;
        public double distance;

        public DetectedTag (int id, double angle, double distance) {
            this.id = id;
            this.angle = angle;
            this.distance = distance;
        }

        //Sort by angle
        @Override
        public int compareTo(DetectedTag o) {
            if (angle < o.angle) {
                return -1;
            } else if (angle > o.angle) {
                return 1;
            } else {
                return 0;
            }
        }
    }

    public static class Pose2D {
        Point xy;
        public double angle;

        public Pose2D(Point xy, double angle) {
            this.xy = xy;
            this.angle = angle;
        }
    }

    public static class Point {
        public double x;
        public double y;

        public Point(double x, double y) {
            this.x = x;
            this.y = y;
        }

        double distance(Point p) {
            return sqrt((p.x-this.x) * (p.x-this.x) +  (p.y-this.y) * (p.y-this.y));
        }
    }

    public static class PointPair {
        Point A;
        Point B;

        public PointPair(Point A, Point B) {
            this.A = A;
            this.B = B;
        }
    }

    public static class Result {
        Pose2D observerPose;
        Double rms;
        int iterations;

        public Result( Pose2D observerPose, Double rms, int iterations) {
            this.observerPose = observerPose;
            this.rms = rms;
            this.iterations = iterations;
        }
    }
    
}

import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

import java.io.IOException;

/**
 * Created by Nusrat on 5/20/2017.
 */

public class Main {
    public static void main(String[] args) throws IOException {

        DoubleMatrix Y, R, X, Theta, kx;

        Y = DoubleMatrix.loadCSVFile("movies.csv");
        R = DoubleMatrix.loadCSVFile("R.csv");
        X = DoubleMatrix.loadCSVFile("X.csv");
        kx = DoubleMatrix.loadCSVFile("kx.csv");
        Theta = DoubleMatrix.loadCSVFile("Theta.csv");


     /*   int num_users = 40;
        int num_movies = 50;
        int num_features = 10;
*/

        int num_users = Theta.rows;
        int num_movies = X.rows;
        int num_features = X.columns;


        X = X.get(new IntervalRange(0, num_movies), new IntervalRange(0,num_features));
        Theta = Theta.get(new IntervalRange(0, num_users), new IntervalRange(0,num_features));
        Y = Y.get(new IntervalRange(0, num_movies), new IntervalRange(0,num_users));
        R = R.get(new IntervalRange(0, num_movies), new IntervalRange(0,num_users));

        DoubleMatrix params = DoubleMatrix.concatVertically(X.reshape(num_movies*num_features,1)
                ,Theta.reshape(num_users*num_features,1));
        params.reshape(num_movies*num_features+num_users*num_features,1);

        long start = System.nanoTime();
        Ml.gradientDescentOptim(new ColFilCostAndGrad(params, Y, R, 1.5, num_users, num_movies, num_features), 2, 0.0001);
        System.out.println(System.nanoTime()-start);




    /*    DoubleMatrix initial_centroids = new DoubleMatrix(3,2,3,6,8,3,2,5);
        int K = 3;
        int max_iters = 15;
        Ml.k_means(kx,initial_centroids,max_iters);*/
        //KMeansCost cost = new KMeansCost(kx,);
    }


}

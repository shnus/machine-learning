import org.jblas.DoubleMatrix;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * Created by Nusrat on 5/21/2017.
 */
public class Ml {
    static void show(DoubleMatrix x){
        System.out.println(x);
    }

    static void save(DoubleMatrix a, String fileName){
        try (PrintWriter writer = new PrintWriter(new FileWriter(fileName))) {
            for(int i = 0; i < a.rows; i++) {
                for (int j = 0; j < a.columns-1; j++){
                    writer.print(a.get(i,j)+", ");
                }
                writer.println(a.get(i,a.columns-1));
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    static void gradientDescent(CostAndGrad cost, int iter, double alpha){

        DoubleMatrix grad;
        DoubleMatrix params_prev;
        DoubleMatrix params;


        for(int i = 0; i < iter; i++){

            grad = cost.getGrad();
            params_prev = cost.getParams();

            params = params_prev.subi(grad.mul(alpha));

            cost.setParams(params);
        }
    }



    static void gradientDescentOptim(CostAndGrad cost, int iter, double alpha){

        DoubleMatrix grad;
        DoubleMatrix params_prev;
        DoubleMatrix params;
        DoubleMatrix v;
        DoubleMatrix prev;

        prev = DoubleMatrix.zeros(cost.getParams().rows, cost.getParams().columns);

        System.out.println(prev);
        for(int i = 0; i < iter; i++){

            grad = cost.getGrad();
            params_prev = cost.getParams();
            System.out.println(prev);
            v = prev.muli(0.9).addi(grad.mul(alpha));
            params = params_prev.subi(v);

            cost.setParams(params);

            prev = prev.copy(v);
            System.out.println(prev);
        }
    }

    private static DoubleMatrix cK;
    public static void k_means(DoubleMatrix X, DoubleMatrix initial_centroids, int max_iters){
        int m = X.rows;
        int n = X.columns;
        int K = initial_centroids.rows;
        DoubleMatrix centroids = initial_centroids;
        DoubleMatrix prev_centrois = centroids;
        DoubleMatrix idx = DoubleMatrix.zeros(m,1);

        KMeansCost kMeansCost = new KMeansCost(X, 0,  X.rows, K);
        Double cost;
        DoubleMatrix c = new DoubleMatrix();

        for (int i = 0; i<max_iters; i++){
            idx = findClosestCentroids(X, centroids);;
            //System.out.println(idx);
            centroids = computeCentroids(X, idx, K);
            System.out.println(centroids);
            c.copy(centroids);
            cost = kMeansCost.solveCost(DoubleMatrix.concatVertically(idx,c.reshape(c.rows*c.columns,1)));
            System.out.println(cost);
        }

    }

    public static DoubleMatrix findClosestCentroids(DoubleMatrix X, DoubleMatrix centroids) {
        int K = centroids.rows;
        DoubleMatrix idx = DoubleMatrix.zeros(X.rows, 1);
        cK = DoubleMatrix.zeros(K,1);
        int kk;
        for(int i = 0; i<X.rows; i++){
            DoubleMatrix dist = DoubleMatrix.zeros(K,1);
            for (int j = 0; j<K; j++){
                dist.put(j,0,X.getRow(i).squaredDistance(centroids.getRow(j)));
            }
            kk = dist.argmin();
            idx.put(i,0,kk);
            cK.put(kk,cK.get(kk)+1);
        }
        return idx;
    }

    private static DoubleMatrix computeCentroids(DoubleMatrix X, DoubleMatrix idx, int K) {
        int m = X.rows;
        int n = X.columns;
        DoubleMatrix centroids = DoubleMatrix.zeros(K,n);
        DoubleMatrix index;
        DoubleMatrix xx;

        for (int i = 0; i<K; i++){
            index = idx.eq(i);
            xx = X.mulColumnVector(index);
            xx = xx.columnSums().divi(cK.get(i));
            centroids.putRow(i, xx);
        }
        return centroids;
    }

}

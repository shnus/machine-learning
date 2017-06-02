import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Nusrat on 5/27/2017.
 */
public class KMeansCost implements CostAndGrad
{
    private double J;
    List<Double> JHistory;
    private DoubleMatrix grad;
    private DoubleMatrix prevGrad;
    private DoubleMatrix X;
    private double lambda;
    private int num_c;
    private int num_k;


    public KMeansCost(DoubleMatrix X, double lambda,
                             int num_c, int num_k){
        JHistory = new ArrayList<>();
        this.X = X;
        this.lambda = lambda;
        this.num_c = num_c;
        this.num_k = num_k;
    }

    Double solveCost(DoubleMatrix params){
        DoubleMatrix c = params.get(new IntervalRange(0,num_c), new IntervalRange(0,1));

        DoubleMatrix k = params.get(new IntervalRange(num_c, params.rows), new IntervalRange(0,1));
        k.reshape(num_k,X.columns);

        Double J = 0.0;
        DoubleMatrix Jm;


        for (int i=0; i<X.rows; i++) {
            J += X.getRow(i).squaredDistance(k.getRow((int) c.get(i)));
        }
        J /= num_c;
        // Solving cost function
        /*Jm = X.sub()
        for (int i=0; i<Jm.rows; i++) {
            for (int j = 0; j < Jm.columns; j++) {
                J+=Math.pow(Jm.get(i,j),2);
            }
        }
        J*=0.5;*/
        JHistory.add(J);
        return J;
    }

    @Override
    public double getJ() {


        return 0;
    }

    @Override
    public DoubleMatrix getGrad() {
        return null;
    }

    @Override
    public DoubleMatrix getPrevGrad() {
        return null;
    }

    @Override
    public DoubleMatrix getParams() {
        return null;
    }

    @Override
    public void setParams(DoubleMatrix params) {

    }
}

import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Nusrat on 5/22/2017.
 */
public class ColFilCostAndGrad implements CostAndGrad {
    private double J;
    List<Double> JHistory;
    private DoubleMatrix grad;
    private DoubleMatrix prevGrad;
    private DoubleMatrix params;
    private DoubleMatrix Y;
    private DoubleMatrix R;
    private double lambda;
    private int num_users;
    private int num_movies;
    private int num_features;
    private int data_i;

    @Override
    public double getJ() {
        return J;
    }

    @Override
    public DoubleMatrix getGrad() {
        return grad;
    }

    @Override
    public DoubleMatrix getPrevGrad() {
        return prevGrad;
    }

    @Override
    public DoubleMatrix getParams() {
        return params;
    }

    @Override
    public void setParams(DoubleMatrix params) {
        this.params = params;
        cfCostFunction();
    }

        public ColFilCostAndGrad(DoubleMatrix params, DoubleMatrix Y, DoubleMatrix R, double lambda,
                                 int num_users, int num_movies, int num_features){
            JHistory = new ArrayList<>();
            this.params = params;
            this.Y = Y;
            this.R = R;
            this.lambda = lambda;
            this.num_movies = num_movies;
            this.num_users = num_users;
            this.num_features = num_features;
            cfCostFunction();

        }

    private void cfCostFunction(){

        DoubleMatrix X = params.get(new IntervalRange(0,num_movies*num_features), new IntervalRange(0,1));
        X.reshape(num_movies, num_features);

        DoubleMatrix Theta = params.get(new IntervalRange(num_movies*num_features, params.rows), new IntervalRange(0,1));
        Theta.reshape(num_users, num_features);

        Double J = 0.0;
        DoubleMatrix X_grad = DoubleMatrix.zeros(X.rows, X.columns);
        DoubleMatrix Theta_grad = DoubleMatrix.zeros(Theta.rows, Theta.columns);
        DoubleMatrix Jm;

        // Solving cost function
        Jm = X.mmul(Theta.transpose()).sub(Y).muli(R);
        for (int i=0; i<Jm.rows; i++) {
            for (int j = 0; j < Jm.columns; j++) {
                J+=Math.pow(Jm.get(i,j),2);
            }
        }
        J*=0.5;

        // regularization for cost function
        double theta_reg = 0;
        for(int i=0; i<Theta.length; i++){
            theta_reg+=Math.pow(Theta.get(i),2);
        }
        theta_reg *= lambda/2;

        double x_reg = 0;
        for(int i=0; i<X.length; i++){
            x_reg+=Math.pow(X.get(i),2);
        }
        x_reg *= lambda/2;

        J += theta_reg + x_reg;

        // Solving gradients with regularization
        X_grad = Jm.mmul(Theta).add(X.mul(lambda));
        Theta_grad = Jm.transpose().mmul(X).add(Theta.mul(lambda));

        //System.out.println(J);

        DoubleMatrix grad = DoubleMatrix.concatVertically(X_grad.reshape(num_movies*num_features,1)
                ,Theta_grad.reshape(num_users*num_features,1));
        grad.reshape(num_movies*num_features+num_users*num_features,1);

        this.J = J;
        JHistory.add(J);
        this.prevGrad = this.grad;
        this.grad = grad;
    }
}

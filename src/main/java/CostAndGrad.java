import org.jblas.DoubleMatrix;

/**
 * Created by Nusrat on 5/23/2017.
 */
public interface CostAndGrad {
    double getJ();
    DoubleMatrix getGrad();
    DoubleMatrix getPrevGrad();
    DoubleMatrix getParams();
    void setParams(DoubleMatrix params);
}

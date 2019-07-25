#include <iostream>
#include <iomanip>
#include <ctime>
#include "math.h"

#include "TasmanianSparseGrid.hpp"

using namespace std;

int main( int argc, const char ** argv ){
/*
 * The purpose of this file is to demonstrate the proper way to call
 * functions from the TASMANIAN library. The 10 examples are for
 * illustrative purposes only.
 *
 */
{
// EXAMPLE 1: integrate: f(x,y) = exp( -x^2 ) * cos( y ) over [-1,1] x [-1,1]
// using classical Smolyak grid with Clenshaw-Curtis points and weights
        TasGrid::TasmanianSparseGrid grid;

        cout << endl << "-------------------------------------------------------------------------------------------------" << endl;
        cout << std::scientific; cout.precision(17);
        cout << "Example 1:  integrate f(x,y) = exp( -x^2 ) * cos( y ), using clenshaw-curtis level nodes" << endl;

        int dim = 2;
        int level = 6;

        grid.makeGlobalGrid( dim, 0, level, TasGrid::type_level,TasGrid::rule_clenshawcurtis );
        double *points = grid.getPoints();
        double *weights = grid.getQuadratureWeights();
        int num_points = grid.getNumPoints();

        double I = 0.0;
        for( int i=0; i<num_points; i++ ){
                double x = points[i*dim];
                double y = points[i*dim+1];
                I += weights[i] * exp( -x*x ) * cos( y );
        }

        double exact = 2.513723354063905e+00;
        double E = fabs( exact - I );
        cout << "          at level: " << level << endl;
        cout << "      the grid has: " << num_points << endl;
        cout << "          integral: " << I << endl;
        cout << "             error: " << E << endl << endl;
        delete[] points;
        delete[] weights;

        level = 7;

        grid.makeGlobalGrid( dim, 0, level, TasGrid::type_level,TasGrid::rule_clenshawcurtis );
        points = grid.getPoints();
        weights = grid.getQuadratureWeights();
        num_points = grid.getNumPoints();

        I = 0.0;
        for( int i=0; i<num_points; i++ ){
                double x = points[i*dim];
                double y = points[i*dim+1];
                I += weights[i] * exp( -x*x ) * cos( y );
        }

        E = fabs( exact - I );
        cout << "          at level: " << level << endl;
        cout << "      the grid has: " << num_points << endl;
        cout << "          integral: " << I << endl;
        cout << "             error: " << E << endl;
        delete[] points;
        delete[] weights;
}

{
// EXAMPLE 2: integrate: f(x,y) = exp( -x^2 ) * cos( y ) over (x,y) in [-5,5] x [-2,3]
// using Gauss-Patterson rules chosen to integrate exactly polynomials of
// total degree up to degree specified by prec
        TasGrid::TasmanianSparseGrid grid;

        cout << endl << "-------------------------------------------------------------------------------------------------" << endl;
        cout << std::scientific; cout.precision(17);
        cout << "Example 2: integrate f(x,y) = exp( -x^2 ) * cos( y ) over [-5,5] x [-2,3] using  Gauss-Patterson nodes" << endl;

        int dim = 2;
        int prec = 20;

        double transform_a[2] = { -5.0, -2.0 };
        double transform_b[2] = {  5.0,  3.0 };

        grid.makeGlobalGrid( dim, 0, prec, TasGrid::type_qptotal, TasGrid::rule_gausspatterson );
        grid.setDomainTransform( transform_a, transform_b );
        double *points = grid.getPoints();
        double *weights = grid.getQuadratureWeights();
        int num_points = grid.getNumPoints();

        double I = 0.0;
        for( int i=0; i<num_points; i++ ){
                double x = points[i*dim];
                double y = points[i*dim+1];
                I += weights[i] * exp( -x*x ) * cos( y );
        }

        double exact = 1.861816427518323e+00;
        double E = fabs( exact - I );
        cout << "          at precision: " << prec << endl;
        cout << "      the grid has: " << num_points << endl;
        cout << "          integral: " << I << endl;
        cout << "             error: " << E << endl << endl;
        delete[] points;
        delete[] weights;

        prec = 40;

        grid.makeGlobalGrid( dim, 0, prec, TasGrid::type_qptotal,TasGrid::rule_gausspatterson );
        grid.setDomainTransform( transform_a, transform_b );
        points = grid.getPoints();
        weights = grid.getQuadratureWeights();
        num_points = grid.getNumPoints();

        I = 0.0;
        for( int i=0; i<num_points; i++ ){
                double x = points[i*dim];
                double y = points[i*dim+1];
                I += weights[i] * exp( -x*x ) * cos( y );
        }

        E = fabs( exact - I );
        cout << "          at precision: " << prec << endl;
        cout << "      the grid has: " << num_points << endl;
        cout << "          integral: " << I << endl;
        cout << "             error: " << E << endl;
        delete[] points;
        delete[] weights;
}

{
// EXAMPLE 3: integrate: f(x,y) = exp( -x^2 ) * cos( y ) over (x,y) in [-5,5] x [-2,3]
// using different rules
        TasGrid::TasmanianSparseGrid grid1, grid2, grid3;

        cout << endl << "-------------------------------------------------------------------------------------------------" << endl;
        cout << std::scientific; cout.precision(2);
        cout << "Example 3: integrate f(x,y) = exp( -x^2 ) * cos( y ) over [-5,5] x [-2,3] using different rules" << endl << endl;

        int dim = 2;
        double exact = 1.861816427518323e+00;
        double transform_a[2] = { -5.0, -2.0 };
        double transform_b[2] = {  5.0,  3.0 };

        cout << setw(10) << " " << setw(20) << "Clenshaw-Curtis" << setw(20) << "Gauss-Legendre" << setw(20) << "Gauss-Patterson" << endl;
        cout << setw(10) << "precision" << setw(10) << "points" << setw(10) << "error" << setw(10) << "points" << setw(10) << "error" << setw(10) << "points" << setw(10) << "error" << endl;

        for( int prec=9; prec<=30; prec += 4 ){
                grid1.makeGlobalGrid( dim, 0, prec, TasGrid::type_qptotal, TasGrid::rule_clenshawcurtis );
                grid1.setDomainTransform( transform_a, transform_b );
                grid2.makeGlobalGrid( dim, 0, prec, TasGrid::type_qptotal, TasGrid::rule_gausslegendre );
                grid2.setDomainTransform( transform_a, transform_b );
                grid3.makeGlobalGrid( dim, 0, prec, TasGrid::type_qptotal, TasGrid::rule_gausspatterson );
                grid3.setDomainTransform( transform_a, transform_b );

                double *points = grid1.getPoints();
                double *weights = grid1.getQuadratureWeights();
                int num_points = grid1.getNumPoints();

                double I = 0.0;
                for( int i=0; i<num_points; i++ ){
                        double x = points[i*dim];
                        double y = points[i*dim+1];
                        I += weights[i] * exp( -x*x ) * cos( y );
                }
                delete[] points;
                delete[] weights;

                double err1 = fabs( I - exact );
                int np1 = num_points;

                points = grid2.getPoints();
                weights = grid2.getQuadratureWeights();
                num_points = grid2.getNumPoints();

                I = 0.0;
                for( int i=0; i<num_points; i++ ){
                        double x = points[i*dim];
                        double y = points[i*dim+1];
                        I += weights[i] * exp( -x*x ) * cos( y );
                }
                delete[] points;
                delete[] weights;

                double err2 = fabs( I - exact );
                int np2 = num_points;

                points = grid3.getPoints();
                weights = grid3.getQuadratureWeights();
                num_points = grid3.getNumPoints();

                I = 0.0;
                for( int i=0; i<num_points; i++ ){
                        double x = points[i*dim];
                        double y = points[i*dim+1];
                        I += weights[i] * exp( -x*x ) * cos( y );
                }
                delete[] points;
                delete[] weights;

                double err3 = fabs( I - exact );
                int np3 = num_points;

                cout << setw(10) << prec << setw(10) << np1 << setw(10) << err1 << setw(10) << np2 << setw(10) << err2 << setw(10) << np3 << setw(10) << err3 << endl;
        }
}

{
// EXAMPLE 4: interpolate: f(x,y) = exp( -x^2 ) * cos( y )
// with a rule that exactly interpolates polynomials of total degree up to
// degree specified by prec
        TasGrid::TasmanianSparseGrid grid;

        cout << endl << "-------------------------------------------------------------------------------------------------" << endl;
        cout << std::scientific; cout.precision(17);
        cout << "Example 4: interpolate f(x,y) = exp( -x^2 ) * cos( y ), using clenshaw-curtis iptotal rule" << endl;

        int dim = 2;
        int out = 1;
        int prec = 10;
        double x_desired[2] = { 0.3, 0.7 };
        double exact = exp( -x_desired[0]*x_desired[0] ) * cos( x_desired[1] );

        grid.makeGlobalGrid( dim, out, prec, TasGrid::type_iptotal, TasGrid::rule_clenshawcurtis );

        int num_points = grid.getNumPoints();
        double *points = grid.getPoints();
        double *vals = new double[num_points];

        for( int i=0; i<num_points; i++ ){
                double x = points[i*dim];
                double y = points[i*dim+1];
                vals[i] = exp( -x*x ) * cos( y );
        }

        grid.loadNeededPoints( vals );

        double res;
        grid.evaluate( x_desired, &res );

        cout << "  using polynomials of total degree up to: " << prec << endl;
        cout << "                             the grid has: " << num_points << " poitns" << endl;
        cout << "                 interpolant at (0.3,0.7): " << res << endl;
        cout << "                                    error: " << fabs( res - exact ) << endl << endl;
        delete[] points;
        delete[] vals;

        prec = 12;
        grid.makeGlobalGrid( dim, out, prec, TasGrid::type_iptotal, TasGrid::rule_clenshawcurtis );

        num_points = grid.getNumPoints();
        points = grid.getPoints();
        vals = new double[num_points];

        for( int i=0; i<num_points; i++ ){
                double x = points[i*dim];
                double y = points[i*dim+1];
                vals[i] = exp( -x*x ) * cos( y );
        }

        grid.loadNeededPoints( vals );

        grid.evaluate( x_desired, &res );

        cout << "  using polynomials of total degree up to: " << prec << endl;
        cout << "                             the grid has: " << num_points << " poitns" << endl;
        cout << "                 interpolant at (0.3,0.7): " << res << endl;
        cout << "                                    error: " << fabs( res - exact ) << endl;
        delete[] points;
        delete[] vals;
}

        // generate 4000 random points between -1 and 1
        srand( clock() );
        double *pnts = new double[4000];
        for( int i=0; i<4000; i++ )  pnts[i] = 2.0 * ((double) rand()) / ( (double) RAND_MAX ) -1.0;

{
// EXAMPLE 5:
// interpolate: f(x1,x2,x3,x4) = exp( -x1^2 ) * cos( x2 ) * exp( -x3^2 ) * cos( x4 )
// with Global and Sequence Leja rules
        int dim = 4;
        int out = 1;
        int prec = 15;
        int start_clock, end_clock;

        // generate the true values of the funciton (for error checking)
        double *tres = new double[1000];
        for( int i=0; i<1000; i++ ){
                double x1 = pnts[i*dim];
                double x2 = pnts[i*dim+1];
                double x3 = pnts[i*dim+2];
                double x4 = pnts[i*dim+3];
                tres[i] = exp( -x1*x1 ) * cos( x2 ) * exp( -x3*x3 ) * cos( x4 );
        }

        TasGrid::TasmanianSparseGrid grid;

        start_clock = clock();
        grid.makeGlobalGrid( dim, out, prec, TasGrid::type_iptotal, TasGrid::rule_leja );
        end_clock = clock();
        int gstage1 = end_clock - start_clock;

        int num_points = grid.getNumPoints();
        double *points = grid.getPoints();
        double *vals = new double[num_points];

        cout << endl << "-------------------------------------------------------------------------------------------------" << endl;
        cout << std::scientific; cout.precision(17);
        cout << "Example 5: interpolate f(x1,x2,x3,x4) = exp( -x1^2 ) * cos( x2 ) * exp( -x3^2 ) * cos( x4 )" << endl;
        cout << "           comparign the performance of Global and Sequence grids with leja nodes" << endl;
        cout << "           using polynomials of total degree up to: " << prec << endl;
        cout << "                                        grids have: " << num_points << " points " << endl;
        cout << "           both grids are evaluated at 1000 random points " << endl << endl;
        for( int i=0; i<num_points; i++ ){
                double x1 = points[i*dim];
                double x2 = points[i*dim+1];
                double x3 = points[i*dim+2];
                double x4 = points[i*dim+3];
                vals[i] = exp( -x1*x1 ) * cos( x2 ) * exp( -x3*x3 ) * cos( x4 );
        }

        start_clock = clock();
        grid.loadNeededPoints( vals );
        end_clock = clock();
        int gstage2 = end_clock - start_clock;

        double *res = new double[1000];

        start_clock = clock();
        #pragma omp parallel for
        for( int i=0; i<1000; i++ ){
                grid.evaluate( &(pnts[i*dim]), &(res[i]) );
        }
        end_clock = clock();
        int gstage3 = end_clock - start_clock;

        double gerr = 0.0;
        for( int i=0; i<1000; i++ )  if ( gerr < fabs( res[i] - tres[i] ) ) gerr = fabs( res[i] - tres[i] );

        TasGrid::TasmanianSparseGrid sgrid;
        start_clock = clock();
        sgrid.makeSequenceGrid( dim, out, prec, TasGrid::type_iptotal, TasGrid::rule_leja );
        end_clock = clock();
        int sstage1 = end_clock - start_clock;

        start_clock = clock();
        sgrid.loadNeededPoints( vals );
        end_clock = clock();
        int sstage2 = end_clock - start_clock;

        start_clock = clock();
        #pragma omp parallel for
        for( int i=0; i<1000; i++ ){
                sgrid.evaluate( &(pnts[i*dim]), &(res[i]) );
        }
        end_clock = clock();
        int sstage3 = end_clock - start_clock;

        cout << setw(15) << "Stage"       << setw(15) << "Global Grid" << setw(15) << "Sequence Grid" << endl;
        cout << setw(15) << "make grid"   << setw(15) << gstage1 << setw(15) << sstage1 << "   cycles" << endl;
        cout << setw(15) << "load values" << setw(15) << gstage2 << setw(15) << sstage2 << "   cycles" << endl;
        cout << setw(15) << "evaluate"    << setw(15) << gstage3 << setw(15) << sstage3 << "   cycles" << endl;

        delete[] points;
        delete[] vals;
        delete[] tres;
        delete[] res;
}

        // generate the true values of the funciton (for error checking)
        double *tres = new double[1000];
        for( int i=0; i<1000; i++ ){
                double x = pnts[i*2];
                double y = pnts[i*2+1];
                tres[i] = exp( -x*x ) * cos( y );
        }

{
// EXAMPLE 6:
// interpolate: f(x,y) = exp( -x^2 ) * cos( y )
// using different refinement schemes
        int dim = 2;
        int out = 1;

        TasGrid::TasmanianSparseGrid grid1, grid2, grid3;

        grid1.makeGlobalGrid( dim, out, 3, TasGrid::type_iptotal, TasGrid::rule_leja );
        grid2.makeGlobalGrid( dim, out, 3, TasGrid::type_iptotal, TasGrid::rule_leja );
        grid3.makeGlobalGrid( dim, out, 3, TasGrid::type_iptotal, TasGrid::rule_leja );

        int num_points = grid1.getNumPoints();
        double *points = grid1.getPoints();
        double *vals = new double[num_points];

        for( int i=0; i<num_points; i++ ){
                double x = points[i*dim];
                double y = points[i*dim+1];
                vals[i] = exp( -x*x ) * cos( y );
        }

        grid1.loadNeededPoints( vals );
        grid2.loadNeededPoints( vals );
        grid3.loadNeededPoints( vals );

        delete[] points;
        delete[] vals;

        cout << endl << "-------------------------------------------------------------------------------------------------" << endl;
        cout << std::scientific; cout.precision(2);
        cout << "Example 6: interpolate: f(x,y) = exp( -x^2 ) * cos( y )" << endl;
        cout << "   using leja nodes and different refinement schemes " << endl;
        cout << "   the error is estimated as the maximum from 1000 random points" << endl << endl;

        cout << setw(10) << " " << setw(20) << "Toral Degree" << setw(20) << "Curved" << setw(20) << "Surplus" << endl;
        cout << setw(10) << "iteration" << setw(10) << "points" << setw(10) << "error" << setw(10) << "points" << setw(10) << "error" << setw(10) << "points" << setw(10) << "error" << endl;

        for( int itr=1; itr<=10; itr++ ){
                grid1.setAnisotropicRefinement( TasGrid::type_iptotal, 10, 0 );

                num_points = grid1.getNumNeeded();
                points = grid1.getNeededPoints();
                vals = new double[num_points];
                for( int i=0; i<num_points; i++ ){
                        double x = points[i*dim];
                        double y = points[i*dim+1];
                        vals[i] = exp( -x*x ) * cos( y );
                }
                grid1.loadNeededPoints( vals );
                delete[] points;
                delete[] vals;
                double err1 = 0.0;
                for( int i=0; i<1000; i++ ){
                        double res;
                        grid1.evaluate( &(pnts[i*dim]), &res );
                        if ( err1 < fabs( res - tres[i] ) ) err1 = fabs( res - tres[i] );
                }

                grid2.setAnisotropicRefinement( TasGrid::type_ipcurved, 10, 0 );
                num_points = grid2.getNumNeeded();
                points = grid2.getNeededPoints();
                vals = new double[num_points];
                for( int i=0; i<num_points; i++ ){
                        double x = points[i*dim];
                        double y = points[i*dim+1];
                        vals[i] = exp( -x*x ) * cos( y );
                }
                grid2.loadNeededPoints( vals );
                delete[] points;
                delete[] vals;
                double err2 = 0.0;
                for( int i=0; i<1000; i++ ){
                        double res;
                        grid2.evaluate( &(pnts[i*dim]), &res );
                        if ( err2 < fabs( res - tres[i] ) ) err2 = fabs( res - tres[i] );
                }

                grid3.setSurplusRefinement( 1.E-10, 0 );
                num_points = grid3.getNumNeeded();
                points = grid3.getNeededPoints();
                vals = new double[num_points];
                for( int i=0; i<num_points; i++ ){
                        double x = points[i*dim];
                        double y = points[i*dim+1];
                        vals[i] = exp( -x*x ) * cos( y );
                }
                grid3.loadNeededPoints( vals );
                delete[] points;
                delete[] vals;
                double err3 = 0.0;
                for( int i=0; i<1000; i++ ){
                        double res;
                        grid3.evaluate( &(pnts[i*dim]), &res );
                        if ( err3 < fabs( res - tres[i] ) ) err3 = fabs( res - tres[i] );
                }

                cout << setw(10) << itr << setw(10) << grid1.getNumPoints() << setw(10) << err1 << setw(10) << grid2.getNumPoints() << setw(10) << err2 << setw(10) << grid3.getNumPoints() << setw(10) << err3 << endl;
        }
}

{
// EXAMPLE 7:
// interpolate: f(x,y) = exp( -x^2 ) * cos( y )
// using localp and semilocalp grids
        int dim = 2;
        int out = 1;
        int depth = 7;

        cout << endl << "-------------------------------------------------------------------------------------------------" << endl;
        cout << std::scientific; cout.precision(2);
        cout << "Example 7: interpolate: f(x,y) = exp( -x^2 ) * cos( y )" << endl;
        cout << "           using localp and semi-localp rules with depth " << depth << endl;
        cout << "           the error is estimated as the maximum from 1000 random points" << endl << endl;

        TasGrid::TasmanianSparseGrid grid;
        grid.makeLocalPolynomialGrid( dim, out, depth, 2, TasGrid::rule_localp );

        int num_points = grid.getNumPoints();
        double *points = grid.getPoints();
        double *vals = new double[num_points];

        for( int i=0; i<num_points; i++ ){
                double x = points[i*dim];
                double y = points[i*dim+1];
                vals[i] = exp( -x*x ) * cos( y );
        }

        grid.loadNeededPoints( vals );

        delete[] points;
        delete[] vals;

        double err1 = 0.0;
        for( int i=0; i<1000; i++ ){
                double res;
                grid.evaluate( &(pnts[i*dim]), &res );
                if ( err1 < fabs( res - tres[i] ) ) err1 = fabs( res - tres[i] );
        }

        grid.makeLocalPolynomialGrid( dim, out, depth, 2, TasGrid::rule_semilocalp );

        num_points = grid.getNumPoints();
        points = grid.getPoints();
        vals = new double[num_points];

        for( int i=0; i<num_points; i++ ){
                double x = points[i*dim];
                double y = points[i*dim+1];
                vals[i] = exp( -x*x ) * cos( y );
        }

        grid.loadNeededPoints( vals );
        delete[] points;
        delete[] vals;

        double err2 = 0.0;
        for( int i=0; i<1000; i++ ){
                double res;
                grid.evaluate( &(pnts[i*dim]), &res );
                if ( err2 < fabs( res - tres[i] ) ) err2 = fabs( res - tres[i] );
        }

        cout << "     Number of points: " << grid.getNumPoints() << endl;
        cout << "     Error for      rule_localp: " << err1 << endl;
        cout << "     Error for  rule_semilocalp: " << err2 << endl;
        cout << " Note: semi-localp wins this competition because the function is very smooth" << endl;
}

        // generate the true values of the funciton (for error checking)
        for( int i=0; i<1000; i++ ){
                double x = pnts[i*2];
                double y = pnts[i*2+1];
                tres[i] = cos( 0.5 * M_PI * x ) * cos( 0.5 * M_PI * y );
        }

{
// EXAMPLE 8:
// interpolate: f(x,y) = cos( 0.5 * pi * x ) * cos( 0.5 * pi * y )
// using localp and semilocalp grids
        int dim = 2;
        int out = 1;
        int depth = 7;

        cout << endl << "-------------------------------------------------------------------------------------------------" << endl;
        cout << std::scientific; cout.precision(2);
        cout << "Example 8: interpolate f(x,y) = cos( 0.5 * pi * x ) * cos( 0.5 * pi * y )" << endl;
        cout << "           using localp and localp-zero rules with depth " << depth << endl;
        cout << "           the error is estimated as the maximum from 1000 random points" << endl << endl;

        TasGrid::TasmanianSparseGrid grid;
        grid.makeLocalPolynomialGrid( dim, out, depth, 2, TasGrid::rule_localp );

        int num_points = grid.getNumPoints();
        double *points = grid.getPoints();
        double *vals = new double[num_points];
        int np1 = num_points; // save for cout purposes

        for( int i=0; i<num_points; i++ ){
                double x = points[i*dim];
                double y = points[i*dim+1];
                vals[i] = cos( 0.5 * M_PI * x ) * cos( 0.5 * M_PI * y );
        }

        grid.loadNeededPoints( vals );
        delete[] points;
        delete[] vals;

        double err1 = 0.0;
        for( int i=0; i<1000; i++ ){
                double res;
                grid.evaluate( &(pnts[i*dim]), &res );
                if ( err1 < fabs( res - tres[i] ) ) err1 = fabs( res - tres[i] );
        }

        grid.makeLocalPolynomialGrid( dim, out, depth-1, 2, TasGrid::rule_localp0 );

        num_points = grid.getNumPoints();
        points = grid.getPoints();
        vals = new double[num_points];

        for( int i=0; i<num_points; i++ ){
                double x = points[i*dim];
                double y = points[i*dim+1];
                vals[i] = cos( 0.5 * M_PI * x ) * cos( 0.5 * M_PI * y );
        }

        grid.loadNeededPoints( vals );
        delete[] points;
        delete[] vals;

        double err2 = 0.0;
        for( int i=0; i<1000; i++ ){
                double res;
                grid.evaluate( &(pnts[i*dim]), &res );
                if ( err2 < fabs( res - tres[i] ) ) err2 = fabs( res - tres[i] );
        }

        cout << " For rule_localp   Number of points: " << np1 << "   Error: " << err1 << endl;
        cout << " For rule_localp0  Number of points: " << grid.getNumPoints() << "   Error: " << err2 << endl;
        cout << " Note: localp-zero wins this competition because the function is zero at the boundary" << endl;
}

        // generate the true values of the funciton (for error checking)
        for( int i=0; i<1000; i++ ){
                double x = pnts[i*2];
                double y = pnts[i*2+1];
                tres[i] = exp( -x ) / ( 1.0 + 100.0 * exp( -10.0 * y ) );
        }

{
// EXAMPLE 9:
// interpolate: f(x,y) = exp( -x ) / ( 1 + 100 * exp( -10 * y ) )
// using different refinement schemes
        int dim = 2;
        int out = 1;
        int depth = 2;
        double tol = 1.E-5;

        TasGrid::TasmanianSparseGrid grid1, grid2;

        grid1.makeLocalPolynomialGrid( dim, out, depth, -1, TasGrid::rule_localp );
        int num_points = grid1.getNumPoints();
        double *points = grid1.getPoints();
        double *vals = new double[num_points];
        for( int i=0; i<num_points; i++ ){
                double x = points[i*dim];
                double y = points[i*dim+1];
                vals[i] = exp( -x ) / ( 1.0 + 100.0 * exp( -10.0 * y ) );
        }
        grid1.loadNeededPoints( vals );

        grid2.makeLocalPolynomialGrid( dim, out, depth, -1, TasGrid::rule_localp );
        grid2.loadNeededPoints( vals );

        delete[] points;
        delete[] vals;

        cout << endl << "-------------------------------------------------------------------------------------------------" << endl;
        cout << std::scientific; cout.precision(2);
        cout << "Example 9: interpolate f(x,y) = exp( -x ) / ( 1 + 100 * exp( -10 * y ) )" << endl;
        cout << "   the error is estimated as the maximum from 1000 random points" << endl;
        cout << "   tolerance is set at 1.E-5 and maximal order polynomials are used" << endl << endl;

        cout << setw(10) << " " << setw(20) << "Classic" << setw(20) << "FDS" << endl;
        cout << setw(10) << "iteration" << setw(10) << "points" << setw(10) << "error" << setw(10) << "points" << setw(10) << "error" << endl;

        for( int itr=1; itr <= 7; itr++ ){
                grid1.setSurplusRefinement( tol, TasGrid::refine_classic );
                num_points = grid1.getNumNeeded();
                points = grid1.getNeededPoints();
                vals = new double[num_points];
                for( int i=0; i<num_points; i++ ){
                        double x = points[i*dim];
                        double y = points[i*dim+1];
                        vals[i] = exp( -x ) / ( 1.0 + 100.0 * exp( -10.0 * y ) );
                }
                grid1.loadNeededPoints( vals );

                delete[] points;
                delete[] vals;
                double err1 = 0.0;
                for( int i=0; i<1000; i++ ){
                        double res;
                        grid1.evaluate( &(pnts[i*dim]), &res );
                        if ( err1 < fabs( res - tres[i] ) ) err1 = fabs( res - tres[i] );
                }

                grid2.setSurplusRefinement( tol, TasGrid::refine_fds );

                num_points = grid2.getNumNeeded();
                points = grid2.getNeededPoints();
                vals = new double[num_points];
                for( int i=0; i<num_points; i++ ){
                        double x = points[i*dim];
                        double y = points[i*dim+1];
                        vals[i] = exp( -x ) / ( 1.0 + 100.0 * exp( -10.0 * y ) );
                }
                grid2.loadNeededPoints( vals );
                delete[] points;
                delete[] vals;
                double err2 = 0.0;
                for( int i=0; i<1000; i++ ){
                        double res;
                        grid2.evaluate( &(pnts[i*dim]), &res );
                        if ( err2 < fabs( res - tres[i] ) ) err2 = fabs( res - tres[i] );
                }

                cout << setw(10) << itr << setw(10) << grid1.getNumPoints() << setw(10) << err1 << setw(10) << grid2.getNumPoints() << setw(10) << err2 << endl;
        }
}

{
// EXAMPLE 10:
// interpolate: f(x,y) = exp( -x ) / ( 1 + 100 * exp( -10 * y ) )
// using local polynomails and wavelets
        int dim = 2;
        int out = 1;
        double tol = 1.E-5;

        TasGrid::TasmanianSparseGrid grid1, grid2;

        grid1.makeLocalPolynomialGrid( dim, out, 3, 1, TasGrid::rule_localp );
        int num_points = grid1.getNumPoints();
        double *points = grid1.getPoints();
        double *vals = new double[num_points];
        for( int i=0; i<num_points; i++ ){
                double x = points[i*dim];
                double y = points[i*dim+1];
                vals[i] = exp( -x ) / ( 1.0 + 100.0 * exp( -10.0 * y ) );
        }
        grid1.loadNeededPoints( vals );

        delete[] points;
        delete[] vals;

        grid2.makeWaveletGrid( dim, out, 1, 1 );
        num_points = grid2.getNumPoints();
        points = grid2.getPoints();
        vals = new double[num_points];
        for( int i=0; i<num_points; i++ ){
                double x = points[i*dim];
                double y = points[i*dim+1];
                vals[i] = exp( -x ) / ( 1.0 + 100.0 * exp( -10.0 * y ) );
        }
        grid2.loadNeededPoints( vals );

                        TasGrid::TasmanianSparseGrid gridc;
                        gridc.copyGrid( &grid2 );
                        grid2.copyGrid( &gridc );

        delete[] points;
        delete[] vals;

        cout << endl << "-------------------------------------------------------------------------------------------------" << endl;
        cout << std::scientific; cout.precision(2);
        cout << "Example 10: interpolate f(x,y) = exp( -x ) / ( 1 + 100 * exp( -10 * y ) )" << endl;
        cout << "   the error is estimated as the maximum from 1000 random points" << endl;
        cout << "   using local polynomials and wavelets" << endl << endl;


        cout << setw(10) << " " << setw(20) << "Polynomials" << setw(20) << "Wavelets" << endl;
        cout << setw(10) << "iteration" << setw(10) << "points" << setw(10) << "error" << setw(10) << "points" << setw(10) << "error" << endl;

        for( int itr=1; itr <= 8; itr++ ){
                grid1.setSurplusRefinement( tol, TasGrid::refine_fds );
                num_points = grid1.getNumNeeded();
                points = grid1.getNeededPoints();
                vals = new double[num_points];
                for( int i=0; i<num_points; i++ ){
                        double x = points[i*dim];
                        double y = points[i*dim+1];
                        vals[i] = exp( -x ) / ( 1.0 + 100.0 * exp( -10.0 * y ) );
                }
                grid1.loadNeededPoints( vals );
                delete[] points;
                delete[] vals;
                double err1 = 0.0;
                for( int i=0; i<1000; i++ ){
                        double res;
                        grid1.evaluate( &(pnts[i*dim]), &res );
                        if ( err1 < fabs( res - tres[i] ) ) err1 = fabs( res - tres[i] );
                }

                grid2.setSurplusRefinement( tol, TasGrid::refine_fds );
                num_points = grid2.getNumNeeded();
                points = grid2.getNeededPoints();
                vals = new double[num_points];
                for( int i=0; i<num_points; i++ ){
                        double x = points[i*dim];
                        double y = points[i*dim+1];
                        vals[i] = exp( -x ) / ( 1.0 + 100.0 * exp( -10.0 * y ) );
                }
                grid2.loadNeededPoints( vals );

                delete[] points;
                delete[] vals;
                double err2 = 0.0;
                for( int i=0; i<1000; i++ ){
                        double res;
                        grid2.evaluate( &(pnts[i*dim]), &res );
                        if ( err2 < fabs( res - tres[i] ) ) err2 = fabs( res - tres[i] );
                }

                cout << setw(10) << itr << setw(10) << grid1.getNumPoints() << setw(10) << err1 << setw(10) << grid2.getNumPoints() << setw(10) << err2 << endl;
        }
}

        cout << endl << "-------------------------------------------------------------------------------------------------" << endl << endl;

        delete[] pnts;
        delete[] tres;

        return 0;
}

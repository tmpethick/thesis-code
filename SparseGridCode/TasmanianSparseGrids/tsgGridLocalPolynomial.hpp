/*
 * Copyright (c) 2017, Miroslav Stoyanov
 *
 * This file is part of
 * Toolkit for Adaptive Stochastic Modeling And Non-Intrusive ApproximatioN: TASMANIAN
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
 *    and the following disclaimer in the documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
 *    or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * UT-BATTELLE, LLC AND THE UNITED STATES GOVERNMENT MAKE NO REPRESENTATIONS AND DISCLAIM ALL WARRANTIES, BOTH EXPRESSED AND IMPLIED.
 * THERE ARE NO EXPRESS OR IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, OR THAT THE USE OF THE SOFTWARE WILL NOT INFRINGE ANY PATENT,
 * COPYRIGHT, TRADEMARK, OR OTHER PROPRIETARY RIGHTS, OR THAT THE SOFTWARE WILL ACCOMPLISH THE INTENDED RESULTS OR THAT THE SOFTWARE OR ITS USE WILL NOT RESULT IN INJURY OR DAMAGE.
 * THE USER ASSUMES RESPONSIBILITY FOR ALL LIABILITIES, PENALTIES, FINES, CLAIMS, CAUSES OF ACTION, AND COSTS AND EXPENSES, CAUSED BY, RESULTING FROM OR ARISING OUT OF,
 * IN WHOLE OR IN PART THE USE, STORAGE OR DISPOSAL OF THE SOFTWARE.
 */

#ifndef __TASMANIAN_SPARSE_GRID_LPOLY_HPP
#define __TASMANIAN_SPARSE_GRID_LPOLY_HPP

#include "tsgEnumerates.hpp"
#include "tsgIndexSets.hpp"
#include "tsgIndexManipulator.hpp"
#include "tsgGridCore.hpp"
#include "tsgRuleLocalPolynomial.hpp"

namespace TasGrid{

class GridLocalPolynomial : public BaseCanonicalGrid{
public:
        GridLocalPolynomial();
        GridLocalPolynomial( const GridLocalPolynomial &pwpoly );
        ~GridLocalPolynomial();

        void write( std::ofstream &ofs ) const;
        void read( std::ifstream &ifs );

        void makeGrid( int cnum_dimensions, int cnum_outputs, int depth, int corder, TypeOneDRule crule );
        void copyGrid( const GridLocalPolynomial *pwpoly );

        int getNumDimensions() const;
        int getNumOutputs() const;
        TypeOneDRule getRule() const;
        int getOrder() const;

        int getNumLoaded() const;
        int getNumNeeded() const;
        int getNumPoints() const;

        double* getLoadedPoints() const;
        double* getNeededPoints() const;
        double* getPoints() const;

        double* getQuadratureWeights() const;
        double* getInterpolationWeights( const double x[] ) const;

        void loadNeededPoints( const double *vals );

        void evaluate( const double x[], double y[] ) const;
        void integrate( double q[] ) const;

        void setSurplusRefinement( double tolerance, TypeRefinement criteria, int output = -1 );
        void clearRefinement();
        int removePointsBySurplus( double tolerance, int output = -1 ); // returns the number of points kept

        double* evalHierarchicalFunctions( const double x[] ) const;
        void setHierarchicalCoefficients( const double c[] );

        const double* getSurpluses() const;
        const int* getPointIndexes() const;
        const int* getNeededIndexes() const;

protected:
        void reset( bool clear_rule = true );

        void buildTree();

        void recomputeSurpluses();

        double evalBasisRaw( const int point[], const double x[] ) const;
        double evalBasisSupported( const int point[], const double x[], bool &isSupported ) const;

        double* getBasisIntegrals() const;

        double* getNormalization() const;

        int* buildUpdateMap( double tolerance, TypeRefinement criteria, int output ) const;

        bool addParent( const int point[], int direction, GranulatedIndexSet *destination, IndexSet *exclude ) const;
        void addChild( const int point[], int direction, GranulatedIndexSet *destination, IndexSet *exclude )const;

private:
        int num_dimensions, num_outputs, order, top_level;

        double *surpluses;

        IndexSet *points;
        IndexSet *needed;

        StorageSet *values;
        int *parents;

        // three for evaluation
        int num_roots, *roots;
        int *pntr, *indx;

        BaseRuleLocalPolynomial *rule;
        RuleLocalPolynomial rpoly;
        RuleSemiLocalPolynomial rsemipoly;
        RuleLocalPolynomialZero rpoly0;
        RuleLocalPolynomialConstant rpolyc;
};

}

#endif

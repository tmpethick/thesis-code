function  [ lGrid, points ] = tsgMakeWavelet( sGridName, iDim, iOut, iDepth, iOrder, mTransformAB )
%
% [ lGrid, points ] = tsgMakeWavelet( sGridName, iDim, iOut, iDepth, iOrder, mTransformAB )
%
% creates a new sparse grid using a sequence rule
%
% INPUT:
%
% sGridName: the name of the grid, give it a string name, 
%            i.e. 'myGrid' or '1' or 'pi314'
%            DO NOT LEAVE THIS EMPTY
%
% iDim: (integer, positive)
%       the number of inputs
%
% iOut: (integer, non-negative)
%       the number of outputs
%
% iDepth: (integer non-negative)
%       controls the density of the grid, i.e., the number of levels to use
%
% iOrder: (integer must be 1 or 3)
%         note that only wavelets of order 1 and 3 are implemented
%
% mTransformAB: (optional matrix of size iDim x 2)
%               for all but gauss-laguerre and gauss-hermite grids, the
%               transform specifies the lower and upper bound of the domain
%               in each direction. For gauss-laguerre and gauss-hermite
%               grids, the transform gives the a and b parameters that
%               change the weight to 
%               exp( -b ( x - a ) )  and  exp( -b ( x - a )^2 )
%
% OUTPUT:
%
% lGrid: list containing information about the sparse grid, can be used to
%        call other functions
%
% points: (optional) the points of the grid in an array of dimension [ num_poits, dim ]
%
% [ lGrid, points ] = tsgMakeWavelet( sGridName, iDim, iOut, iDepth, iOrder, mTransformAB )
%

% create lGrid object
lGrid.sName = sGridName;
lGrid.iDim  = iDim;
lGrid.iOut  =  iOut;
lGrid.sType = 'wavelet';

% check for conflict with tsgMakeQuadrature
if ( strcmp( sGridName, '' ) )
    error('sGridName cannot be empty');
end

% generate filenames
[ sFiles, sTasGrid ] = tsgGetPaths();
[ sFileG, sFileX, sFileV, sFileO, sFileW, sFileC ] = tsgMakeFilenames( lGrid.sName );

sCommand = [sTasGrid,' -makewavelet'];

sCommand = [ sCommand, ' -gridfile ',   sFileG];
sCommand = [ sCommand, ' -dimensions ', num2str(lGrid.iDim)];
sCommand = [ sCommand, ' -outputs ',    num2str(lGrid.iOut)];
sCommand = [ sCommand, ' -depth ',      num2str(iDepth)];
sCommand = [ sCommand, ' -order ',      num2str(iOrder)];

% set the domain transformation
if ( exist('mTransformAB') && (max( size( mTransformAB )) ~= 0) )
    if ( size( mTransformAB, 2 ) ~= 2 )
        error(' mTransformAB must be a matrix with 2 columns');
    end
    if ( size( mTransformAB, 1 ) ~= lGrid.iDim )
        error(' mTransformAB must be a matrix with iDim number of rows');
    end
    tsgWriteMatrix( sFileV, mTransformAB );
    lGlean.sFileV = 1;
    sCommand = [ sCommand, ' -tf ',sFileV];
end

% read the points for the grid
if ( nargout > 1 )
    sCommand = [ sCommand, ' -of ',sFileO];
    lGlean.sFileO = 1;
end

[status, cmdout] = system(sCommand);

if ( max( size( findstr( 'ERROR', cmdout ) ) ) ~= 0 )
    disp(cmdout);
    error('The tasgrid execurable returned an error, see above');
    return;
else
    if ( ~isempty(cmdout) )
        fprintf(1,['WARNING: Command had non-empty output:\n']);
        disp(cmdout);
    end
    if ( nargout > 1 )
        points = tsgReadMatrix( sFileO );
    end
end

if ( exist( 'lClean' ) )
    tsgCleanTempFiles( lGrid, lClean );
end

end
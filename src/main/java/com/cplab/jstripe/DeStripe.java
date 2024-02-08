/**
 * De-stripes light sheet images using either the Kurst et al. method or
 * a different method that is robust to local changes in intensity as well as
 * somewhat angled stripes.

 * Original algorithm based on:
 * https://www.sciencedirect.com/science/article/pii/S0092867420301094
 * https://github.com/chunglabmit/pystripe
 */
package com.cplab.jstripe;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;
import java.util.HashMap;
import java.util.List;

import io.scif.config.SCIFIOConfig;
import io.scif.services.DatasetIOService;
import net.imagej.axis.Axes;
import net.imglib2.*;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.realtransform.InverseRealTransform;
import net.imglib2.type.logic.BitType;
import org.scijava.app.StatusService;
import org.scijava.command.Command;
import org.scijava.command.Previewable;
import org.scijava.Initializable;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.command.DynamicCommand;
import org.scijava.ui.UIService;
import org.scijava.module.MutableModuleItem;

import io.scif.img.SCIFIOImgPlus;

import net.imagej.Dataset;
import net.imagej.DatasetService;
import net.imagej.ImageJ;
import net.imagej.display.ImageDisplayService;
import net.imagej.axis.AxisType;
import net.imagej.ops.OpService;
import net.imglib2.realtransform.Scale2D;
import net.imglib2.view.RandomAccessibleOnRealRandomAccessible;
import net.imglib2.realtransform.RealViews;
import net.imglib2.interpolation.randomaccess.NLinearInterpolatorFactory;
import net.imglib2.type.NativeType;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.RealType;
import net.imglib2.loops.LoopBuilder;
import net.imagej.ImgPlus;
import net.imglib2.realtransform.RealTransformRandomAccessible;
import net.imglib2.util.Util;
import net.imglib2.view.Views;
import net.imglib2.view.HyperSlice;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.img.array.ArrayImg;


/** Helper class for de-striping.

 * Tracks the size and indices of the input dataset so we can fix some dimensions.

 * Also caches images, image grid sampler, and scale transformation for re-use.
 */
final class DeStripeHelper {

    private final int indexT;
    private final int indexC;
    private final int indexX;
    private final int indexY;
    private final int indexZ;

    private final long sizeT;
    private final long sizeC;
    private final long sizeX;
    private final long sizeY;

    // we only fix T, C, and Z dim to a single slice so X and Y are not important
    private final long[] sliceIdx;
    // We need to know the size of the X and Y range for when we fix the others
    private final long[] maxSlice;
    private final long[] minSlice;
    // The dim index in the dataset of the fixed T, C, and Z dim if any are present
    private final ArrayList<Integer> imgSlice;

    private final Map <Long, ArrayList<CellGrid>> gridMap;
    private final Map <Long, ArrayList<ArrayImg<FloatType, ?>>> imgMap;
    private final Map <Long, ArrayList<Scale2D>> scaleMap;

    private final ArrayImgFactory<FloatType> imgFactory;

    final public NLinearInterpolatorFactory<FloatType> interpolatorFactory;

    public DeStripeHelper(final Dataset dataset) {
        indexT = dataset.dimensionIndex(Axes.TIME);
        indexC = dataset.dimensionIndex(Axes.CHANNEL);
        indexX = dataset.dimensionIndex(Axes.X);
        indexY = dataset.dimensionIndex(Axes.Y);
        indexZ = dataset.dimensionIndex(Axes.Z);

        if (indexX < 0 || indexY < 0) {
            throw new ArrayIndexOutOfBoundsException("No X or Y dimension found");
        }

        sizeT = dataset.dimension(Axes.TIME);
        sizeC = dataset.dimension(Axes.CHANNEL);
        sizeX = dataset.dimension(Axes.X);
        sizeY = dataset.dimension(Axes.Y);

        sliceIdx = new long[dataset.numDimensions()];
        sliceIdx[indexX] = 0;
        sliceIdx[indexY] = 0;

        maxSlice = new long[2];
        minSlice = new long[]{0, 0};
        // the hyper-slice is sorted by unfixed dimensions (X and Y)
        maxSlice[0] = indexY > indexX ? dataset.dimension(Axes.X) - 1 : dataset.dimension(Axes.Y) - 1;
        maxSlice[1] = indexY > indexX ? dataset.dimension(Axes.Y) - 1 : dataset.dimension(Axes.X) - 1;
        imgSlice = new ArrayList<>();

        gridMap = new HashMap<>();
        imgMap = new HashMap<>();
        scaleMap = new HashMap<>();

        imgFactory = new ArrayImgFactory<>(new FloatType());
        interpolatorFactory = new NLinearInterpolatorFactory<>();
    }

    public CellGrid getGrid(int cellX, int cellY) {
        long key = cellX + ((long) cellY >> 32);
        ArrayList<CellGrid> array = gridMap.get(key);

        if (array == null || array.isEmpty()) {
            return new CellGrid(new long[]{sizeX, sizeY}, new int[]{cellX, cellY});
        } else {
            return array.remove(array.size() - 1);
        }
    }

    public void returnGrid(CellGrid grid) {
        long key = grid.cellDimension(0) + ((long) grid.cellDimension(1) >> 32);

        if (gridMap.containsKey(key)) {
            gridMap.get(key).add(grid);
        }
        else {
            ArrayList<CellGrid> array = new ArrayList<>();
            array.add(grid);
            gridMap.put(key, array);
        }
    }

    public ArrayImg<FloatType, ?> getImage(long x, long y) {
        long key = x + (y >> 32);
        ArrayList<ArrayImg<FloatType, ?>> array = imgMap.get(key);

        if (array == null || array.isEmpty()) {
            return imgFactory.create(x, y);
        } else {
            return array.remove(array.size() - 1);
        }
    }

    public void returnImage(ArrayImg<FloatType, ?> img) {
        long key = img.dimension(0) + (img.dimension(1) >> 32);

        if (imgMap.containsKey(key)) {
            imgMap.get(key).add(img);
        }
        else {
            ArrayList<ArrayImg<FloatType, ?>> array = new ArrayList<>();
            array.add(img);
            imgMap.put(key, array);
        }
    }

    public Scale2D getScale(int smallerX, int smallerY) {
        long key = smallerX + ((long) smallerY >> 32);
        ArrayList<Scale2D> array = scaleMap.get(key);

        if (array == null || array.isEmpty()) {
            return new Scale2D(
                    (double) sizeX / smallerX,
                    (double) sizeY / smallerY
            );
        } else {
            return array.remove(array.size() - 1);
        }
    }

    public void returnScale(Scale2D scale, int smallerX, int smallerY) {
        long key = smallerX + ((long) smallerY >> 32);

        if (scaleMap.containsKey(key)) {
            scaleMap.get(key).add(scale);
        }
        else {
            ArrayList<Scale2D> array = new ArrayList<>();
            array.add(scale);
            scaleMap.put(key, array);
        }
    }

    public RandomAccessibleInterval< ? > getCurrentImgSlice(RandomAccessibleInterval< ? > interval) {
        if (imgSlice.isEmpty()) {
            return interval;
        } else {
            int[] fixedDims = imgSlice.stream().mapToInt(Integer::intValue).toArray();
            return Views.interval(new HyperSlice<>(interval, fixedDims, sliceIdx), minSlice, maxSlice);
        }
    }

    public void fixDim(String axis, long value) {
        switch (axis) {
            case "T":
                if (indexT >= 0) {
                    sliceIdx[indexT] = value;
                    imgSlice.add(indexT);
                }
                break;
            case "C":
                if (indexC >= 0) {
                    sliceIdx[indexC] = value;
                    imgSlice.add(indexC);
                }
                break;
            case "Z":
                if (indexZ >= 0) {
                    sliceIdx[indexZ] = value;
                    imgSlice.add(indexZ);
                }
                break;
        }
    }

    public void unfixDim(String axis) {
        switch (axis) {
            case "T":
                if (indexT >= 0) {
                    imgSlice.remove((Object) indexT);
                }
                break;
            case "C":
                if (indexC >= 0) {
                    imgSlice.remove((Object) indexC);
                }
                break;
            case "Z":
                if (indexZ >= 0) {
                    imgSlice.remove((Object) indexZ);
                }
                break;
        }
    }

    public long getSizeT() {
        return sizeT;
    }

    public long getSizeC() {
        return sizeC;
    }
}


@Plugin(type = Command.class, menuPath = "Plugins>SPIM Destriping")
public class DeStripe extends DynamicCommand implements Command, Previewable, Initializable {

    @Parameter
    private UIService ui;

    @Parameter
    private LogService log;

    @Parameter
    private StatusService statusService;

    @Parameter(required = false)
    private ImageDisplayService idService;

    @Parameter
    private DatasetIOService datasetIOService;

    @Parameter
    private DatasetService datasetService;

    @Parameter
    private OpService opService;

    final private String defaultStringFile = "<Use filename below>";

    @Parameter(
            label = "Select open image (or file below)",
            choices = { defaultStringFile },
            persist = false,
            description = "If you want to use an open dataset select it here, otherwise " +
                    "select '" + defaultStringFile + "' and select a file below."
    )
    private String inputImage = defaultStringFile;

    @Parameter(
            label = "Select file from disk (or open dataset above)",
            required = false, persist = false,
            description = "If not using an open dataset, browse to a file to open."
    )
    private File inputFile;

    @Parameter(
            label = "Load whole file into memory",
            description = "When opening a file from disk, whether to load the whole " +
                    "image into memory or load a slice at a time from disk if it's too large."
    )
    private boolean bufferInput = true;

    @Parameter(
            label = "Display file (if opened from disk)",
            description = "Whether to display the input file in a window."
    )
    private boolean displayInput = false;

    @Parameter(
            label = "Display the intermediate calculation images",
            description = "For debugging, whether to display each step of the intermediate computations " +
                    "in a window. WARNING - will open multiple windows."
    )
    private boolean displayCalc = false;

    @Parameter(
            label = "Whether to use the original algorithm of Kirst et al.",
            description = "Whether to use the original algorithm by Kirst et al. The original algorithm " +
                    "is only useful when the stripes are well aligned with an axis and there's overall similar " +
                    "background intensity."
    )
    private boolean originalAlgo = false;

    @Parameter(
            label = "Percentile for voxel illumination [0-100]", min = "0", max = "100",
            description = "Value between 0-100 used to find the \"typical\" intensity value in " +
                    "a rectangle or region."
    )
    private int percentileStripe = 25;

    @Parameter(
            label = "# of X pixels in stripe-parallel rectangle", min = "1", max = "1000",
            description = "The number of pixels in the X direction to construct the rectangles that are " +
                    "parallel to the stripes. I.e. the rectangles used to estimate the intensity along the stripes."
    )
    private int parallelRectX = 150;

    @Parameter(
            label = "# of Y pixels in stripe-parallel rectangle", min = "1", max = "1000",
            description = "The number of pixels in the Y direction to construct the rectangles that are " +
                    "parallel to the stripes. I.e. the rectangles used to estimate the intensity along the stripes."
    )
    private int parallelRectY = 1;

    @Parameter(
            label = "# of X pixels in background rectangle", min = "1", max = "1000",
            description = "The number of pixels in the X direction to construct the background rectangles. " +
                    "I.e. the rectangles used to estimate the intensity of the overall area around each pixel."
    )
    private int backgroundRectX = 200;

    @Parameter(
            label = "# of Y pixels in background rectangle", min = "1", max = "1000",
            description = "The number of pixels in the Y direction to construct the background rectangles. " +
                    "I.e. the rectangles used to estimate the intensity of the overall area around each pixel."
    )
    private int backgroundRectY = 200;

    @Parameter(
            label = "# of X pixels in stripe-perpendicular rectangle", min = "1", max = "1000",
            description = "The number of pixels in the X direction to construct the background rectangles " +
                    "that are perpendicular to the stripes. I.e. the rectangles used to estimate the intensity " +
                    "of the stripe-perpendicular area around each pixel. If not desired, set it the same as " +
                    "the background rectangle, since the minimum of both intensities is used. " +
                    "*Ignored when the original algorithm is used*."
    )
    private int backgroundPerpRectX = 1;

    @Parameter(
            label = "# of Y pixels in stripe-perpendicular rectangle", min = "1", max = "1000",
            description = "The number of pixels in the Y direction to construct the background rectangles " +
                    "that are perpendicular to the stripes. I.e. the rectangles used to estimate the intensity " +
                    "of the stripe-perpendicular area around each pixel. If not desired, set it the same as " +
                    "the background rectangle, since the minimum of both intensities is used. " +
                    "*Ignored when the original algorithm is used*."
    )
    private int backgroundPerpRectY = 150;

    @Parameter(
            label = "Stripe multiplication difference factor", min = "0.0", max = "100.0",
            description = "After computing the difference between the rectangle in the stripe direction " +
                    "and the other rectangles. This difference is multiplied by the given factor."
    )
    private double differenceMultFactor = 1.0;

    @Parameter(
            label = "Stripe difference threshold", min = "0.0",
            description = "After computing the difference between the rectangle in the stripe direction " +
                    "and the other rectangles. Only areas whose (absolute) difference is less than the " +
                    "threshold is fixed. " +
                    "*Ignored when the original algorithm is used*."
    )
    private double differenceThreshold = 3000.0;

    @Parameter(
            label = "Process only current slice (for stack)",
            description = "Only processes the current slice in the stack, if the dataset is already" +
                    " open."
    )
    private boolean currentSlice = true;

    @Parameter(
            label = "Process slice range (or all slices - for stack)",
            description = "Whether to process the whole stack or just a range specified below."
    )
    private boolean doSliceRange = false;

    @Parameter(
            label = "Start slice (if range) - inclusive", min = "0",
            description = "The start slice in the range to process if range is enabled."
    )
    private int startSlice = 0;

    @Parameter(
            label = "Last slice (if range) - inclusive", min = "0",
            description = "The last slice in the range to process if range is enabled."
    )
    private int lastSlice = 0;

    @Parameter(
            label = "Display result",
            description = "Whether to display the processed result in the viewer."
    )
    private boolean displayResult = true;

    @Parameter(
            choices = {"float", "16bit int"}, label = "Data type of saved file",
            description = "The byte format of the output dataset."
    )
    private String outputType;

    @Parameter(
            label = "Save to disk (optional - if provided)", required = false, persist = false,
            description = "Filename to save the processed results, if provided."
    )
    private File outputFile;

    private DeStripeHelper helper;

    @Override
    public void initialize() {
        // update the dataset field with list of open datasets
        ArrayList<String> openDatasets = new ArrayList<>();

        List<Dataset> datasets = datasetService.getDatasets();
        final MutableModuleItem<String> inputImages = getInfo().getMutableInput(
                "inputImage", String.class);

        openDatasets.add(defaultStringFile);
        for (Dataset dataset : datasets) {
            openDatasets.add(dataset.getName());
        }

        inputImages.setChoices(openDatasets);
    }

    @Override
    public void run() {
        Dataset result;
        Dataset dataset;
        SCIFIOImgPlus<?> imgIn;
        SCIFIOImgPlus<?> imgOut;
        final long[] zRange = {0, 0};

        // input dataset
        dataset = getUserDataset();
        if (dataset == null)
            return;
        try {
            // ensure we opened it with scifio
            imgIn = getImgPlus(dataset);
        } catch (final ClassCastException e) {
            log.error(e);
            return;
        }

        try {
            helper = new DeStripeHelper(dataset);
        } catch (final ClassCastException e) {
            log.error(e);
            return;
        }

        // requested range
        computeZExtent(dataset, zRange);

        // dataset for result
        if (outputType.equals("float")) {
            result = create(
                    dataset, new FloatType(),
                    zRange[1] - zRange[0] + 1, imgIn, datasetService
            );
        } else {
            result = create(
                    dataset, new UnsignedShortType(),
                    zRange[1] - zRange[0] + 1, imgIn, datasetService
            );
        }

        imgOut = getImgPlus(result);
        processDataset(dataset, result, imgIn, imgOut, zRange, outputType.equals("float"));

        if (displayResult) {
            ui.show(result);
        }

        saveDataset(result);
    }

    /** Get the z stack range requested by user, returning a valid range
     * for the dataset.
     */
    private void computeZExtent(Dataset dataset, long[] zRange) {
        if (currentSlice) {
            if (idService == null) {
                log.error("No display, but current slice requested");
                return;
            }
            if (dataset.dimensionIndex(Axes.Z) >= 0)
                zRange[0] = zRange[1] = idService.getActivePosition().getLongPosition(dataset.dimensionIndex(Axes.Z));
        } else if (doSliceRange) {
            zRange[0] = startSlice;
            zRange[1] = lastSlice;
        } else {
            zRange[1] = Math.max(dataset.getDepth() - 1, 0);
        }

        zRange[0] = Math.min(Math.max(zRange[0], 0), dataset.getDepth() - 1);
        zRange[1] = Math.min(Math.max(zRange[0], zRange[1]), dataset.getDepth() - 1);
    }

    private Dataset getUserDataset() {
        Dataset dataset = null;

        // if no open dataset specified, open file
        if (inputImage.equals(defaultStringFile)) {
            if (inputFile == null) {
                log.error("No input file provided");
                return null;
            }

            SCIFIOConfig config = new SCIFIOConfig();
            config.enableBufferedReading(bufferInput);
            try {
                dataset = datasetIOService.open(inputFile.getAbsolutePath(), config);
            } catch (final IOException e) {
                log.error(e);
                return null;
            }
            if (displayInput)
                ui.show(dataset);
        } else {
            // otherwise use open dataset
            List<Dataset> datasets = datasetService.getDatasets();
            for (Dataset datasetItem : datasets) {
                if (datasetItem.getName().equals(inputImage)) {
                    dataset = datasetItem;
                    break;
                }
            }
            if (dataset == null)
            {
                log.error(new IllegalArgumentException("Can't find dataset by name"));
                return null;
            }
        }

        return dataset;
    }

    /**
     * Gets a SCIFIOImgPlus image object representing the dataset.
     */
    private static SCIFIOImgPlus<?> getImgPlus(Dataset data) {
        ImgPlus<?> imp = data.getImgPlus();

        if (imp instanceof SCIFIOImgPlus) {
            return  (SCIFIOImgPlus<?>) imp;
        }

        return new SCIFIOImgPlus<>(imp);
    }

    /** Creates a new dataset of the specified type, based on the input dataset
     * and metadata, if provided.
     */
    private static <T extends RealType<T> & NativeType<T>> Dataset create(
            final Dataset d, final T type, long zExtent, SCIFIOImgPlus<?> imgIn,
            DatasetService datasetService)
    {
        final int dimCount = d.numDimensions();
        final long[] dims = new long[d.numDimensions()];
        final AxisType[] axes = new AxisType[dimCount];

        d.dimensions(dims);
        int index = d.dimensionIndex(Axes.Z);
        if (index >= 0)
            dims[index] = zExtent;

        for (int i = 0; i < dimCount; i++) {
            axes[i] = d.axis(i).type();
        }

        Dataset result = datasetService.create(type, dims, "result", axes);
        SCIFIOImgPlus<?> img = getImgPlus(result);
        if (imgIn != null) {
            img.setMetadata(imgIn.getMetadata());
            img.setImageMetadata(imgIn.getImageMetadata());
        }

        return result;
    }

    /** Save the dataset to the file, if requested by user
     */
    private void saveDataset(Dataset dataset) {
        if (outputFile != null && !outputFile.getAbsolutePath().isEmpty()) {
            SCIFIOConfig config = new SCIFIOConfig();
            try {
                datasetIOService.save(dataset, outputFile.getAbsolutePath(), config);
            } catch (final IOException e) {
                log.error(e);
            }
        }
    }

    private void computeCellPercentiles(
            Img<FloatType> inputImg, double percentile, int cellX, int cellY, ArrayImg<FloatType, ?> outImg
    ) {
        CellGrid grid = helper.getGrid(cellX, cellY);
        ArrayImg<FloatType, ?> gridImg = helper.getImage(grid.gridDimension(0), grid.gridDimension(1));
        Scale2D scaleTrans = helper.getScale((int) grid.gridDimension(0), (int) grid.gridDimension(1));

        LoopBuilder.setImages(gridImg, grid.cellIntervals()).multiThreaded().forEachPixel(
                (o, g) -> {
                    o.setReal(Util.percentile(
                            Util.asDoubleArray((RandomAccessibleInterval) Views.interval(inputImg, g)),
                            percentile / 100.)
                    );
                }
        );

        RealRandomAccessible< FloatType > interpolated = Views.interpolate(
                Views.extendBorder( gridImg ), helper.interpolatorFactory);
        RealTransformRandomAccessible<FloatType, InverseRealTransform> scaledUp = RealViews.transform(
                interpolated, scaleTrans );
        RandomAccessibleOnRealRandomAccessible<FloatType> rasterized = Views.raster( scaledUp );

        LoopBuilder.setImages(outImg, Views.interval( rasterized, inputImg )).forEachPixel(
                (o, g) -> o.set(g.get())
        );

        helper.returnGrid(grid);
        helper.returnImage(gridImg);
        helper.returnScale(scaleTrans, (int) grid.gridDimension(0), (int) grid.gridDimension(1));
    }

    private void destripe(Img<FloatType> tmpIOImg) {
        RandomAccessibleInterval<BitType> mask = (RandomAccessibleInterval<BitType>) opService.run(
                "threshold.huang", tmpIOImg);

        ArrayImg<FloatType, ?> imgPara = helper.getImage(tmpIOImg.dimension(0), tmpIOImg.dimension(1));
        computeCellPercentiles(tmpIOImg, percentileStripe, parallelRectX, parallelRectY, imgPara);
        ArrayImg<FloatType, ?> imgBack = helper.getImage(tmpIOImg.dimension(0), tmpIOImg.dimension(1));
        computeCellPercentiles(tmpIOImg, percentileStripe, backgroundRectX, backgroundRectY, imgBack);

        if (displayCalc) {
            ui.show("Stripe-parallel rectangle percentiles", imgPara.copy());
            ui.show("Background rectangle percentiles", imgBack.copy());
        }

        if (originalAlgo) {
            // multiply background by factor
            opService.run(
                    "math.multiply",
                    imgBack, imgBack, new FloatType((float) differenceMultFactor)
            );
            // min of background and parallel rectangles
            LoopBuilder.setImages(imgBack, imgPara).multiThreaded().forEachPixel(
                    (br, f) -> {
                        br.setReal(Math.min(br.getRealFloat(), f.getRealFloat()));
                    }
            );
            // min of pixel value and of the background/parallel rectangles
            LoopBuilder.setImages(imgBack, tmpIOImg).multiThreaded().forEachPixel(
                    (br, i) -> {
                        br.setReal(Math.min(br.getRealFloat(), i.getRealFloat()));
                    }
            );
            if (displayCalc) {
                ui.show("Minimum of background and stripe-parallel rectangles and pixel values.", imgBack.copy());
            }
            // pixel minus the previous min value
            opService.run("math.subtract", tmpIOImg, tmpIOImg, imgBack);
        } else {
            ArrayImg<FloatType, ?> imgPer = helper.getImage(tmpIOImg.dimension(0), tmpIOImg.dimension(1));
            computeCellPercentiles(
                    tmpIOImg, percentileStripe, backgroundPerpRectX, backgroundPerpRectY,
                    imgPer
            );

            if (displayCalc) {
                ui.show("Stripe-perpendicular rectangle percentiles", imgPer.copy());
            }

            // compute stripe/background intensity difference
            LoopBuilder.setImages(imgBack, imgPara, imgPer, mask).multiThreaded().forEachPixel(
                    (br, par, perp, m) -> {
                        if (m.get() && Math.abs(br.getRealFloat() - par.getRealFloat()) < differenceThreshold)
                            br.setReal(
                                    Math.max(
                                            Math.min(br.getRealFloat(), perp.getRealFloat()) - par.getRealFloat(), 0.
                                    ) * differenceMultFactor
                            );
                        else
                            br.setReal(0.0);
                    }
            );

            if (displayCalc) {
                ui.show("Stripe to background intensity difference", imgBack.copy());
            }

            // add the difference to the original images
            LoopBuilder.setImages(imgBack, tmpIOImg).multiThreaded().forEachPixel(
                    (br, i) -> {
                        i.setReal(br.getRealFloat() + i.getRealFloat());
                    }
            );

            helper.returnImage(imgPer);
        }

        helper.returnImage(imgPara);
        helper.returnImage(imgBack);
    }

    private void processDataset(
            final Dataset src, Dataset dst, SCIFIOImgPlus<?> imgIn,
            SCIFIOImgPlus<?> imgOut, long[] zRange, boolean floatOut
    ) {
        statusService.showStatus("De-striping");
        final Img<?> srcImg = imgIn.getImg();
        final Img<?> dstImg = imgOut.getImg();

        if (!Arrays.equals(src.dimensionsAsLongArray(), dst.dimensionsAsLongArray())){
            log.error("Dimensions of src and dst arrays don't match");
            return;
        }

        // image passed in and out during the de-striping process with input and resulted image
        ArrayImg<FloatType, ?> tmpIOImg = helper.getImage(src.dimension(Axes.X), src.dimension(Axes.Y));

        final long sizeT = helper.getSizeT();
        final long sizeC = helper.getSizeC();

        // for progress tracking
        int totalImages = (int) (sizeT * sizeC * (zRange[1] - zRange[0] + 1));
        int count = 0;

        for (long t = 0; t < sizeT; t++) {
            helper.fixDim("T", t);
            for (long c = 0; c < sizeC; c++) {
                helper.fixDim("C", c);
                for (long z = zRange[0]; z <= zRange[1]; z++) {
                    helper.fixDim("Z", z);
//                    logSlice(src, dst, zRange, pos);

                    // start with float image stored in tmpIOImg no matter the input
                    opService.run("convert.float32", tmpIOImg, helper.getCurrentImgSlice(srcImg));

                    destripe(tmpIOImg);

                    // convert to resulting datatype
                    if (floatOut) {
                        LoopBuilder.setImages(
                                (RandomAccessibleInterval<FloatType>) helper.getCurrentImgSlice(dstImg), tmpIOImg
                        ).multiThreaded().forEachPixel(
                                (o, g) -> o.set(g.get())
                        );
                    } else {
                        LoopBuilder.setImages(
                                (RandomAccessibleInterval<UnsignedShortType>) helper.getCurrentImgSlice(dstImg),
                                (RandomAccessibleInterval<UnsignedShortType>) opService.run(
                                        "convert.uint16", tmpIOImg)
                        ).multiThreaded().forEachPixel(
                                (o, g) -> o.set(g.get())
                        );
                    }

                    count++;
                    statusService.showProgress(count, totalImages);

                    helper.unfixDim("Z");
                }
                helper.unfixDim("C");
            }
            helper.unfixDim("T");
        }

        helper.returnImage(tmpIOImg);
    }

    private void logSlice(Dataset src, Dataset dst, long[] zRange, long[] pos) {
        Object[] items = {
                src.numDimensions(), dst.numDimensions(),

                src.dimension(Axes.TIME), src.dimension(Axes.CHANNEL),
                src.dimension(Axes.Z), src.dimension(Axes.X),
                src.dimension(Axes.Y),

                dst.dimension(Axes.TIME), dst.dimension(Axes.CHANNEL),
                dst.dimension(Axes.Z), dst.dimension(Axes.X),
                dst.dimension(Axes.Y),

                src.dimensionIndex(Axes.TIME), src.dimensionIndex(Axes.CHANNEL),
                src.dimensionIndex(Axes.Z), src.dimensionIndex(Axes.X),
                src.dimensionIndex(Axes.Y),

                dst.dimensionIndex(Axes.TIME), dst.dimensionIndex(Axes.CHANNEL),
                dst.dimensionIndex(Axes.Z), dst.dimensionIndex(Axes.X),
                dst.dimensionIndex(Axes.Y),

                String.join(", ", Arrays.toString(zRange)),
                String.join(", ", Arrays.toString(pos))
        };
        String[] strings = Arrays.stream(items).map(Object::toString).toArray(String[]::new);
        log.info(String.join(", ", strings));
    }

    public static void main(final String... args) throws Exception {
        // Create the ImageJ application context with all available services
        final ImageJ ij = new ImageJ();
        ij.launch(args);

        // Launch the "Add Two Datasets" command right away.
        ij.command().run(DeStripe.class, true);
    }

}
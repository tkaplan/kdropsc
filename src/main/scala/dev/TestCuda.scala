package dev

import java.awt.image.{DataBufferByte, BufferedImage}
import java.io._
import java.nio.ByteBuffer
import java.util.Calendar
import javax.imageio.ImageIO

import dev.la.Matrix
import jcuda.driver.JCudaDriver._

import jcuda._
import jcuda.driver._
import jcuda.jcurand._
import jcuda.jcurand.JCurand._
import jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_DEFAULT
import jcuda.runtime._

/**
 * Created by dev on 6/7/15.
 */
object TestCuda {
  private def bindCtx = {
    JCudaDriver.setExceptionsEnabled(true)

    val device = new CUdevice
    // Get and bind to the first device we find
    cuDeviceGet(device, 0)
    val context = new CUcontext
    cuCtxCreate(context, 0, device)

    context
  }

  private def getKernel(kernel: String) = {
    val kernelPath = getClass.getResource(s"/kernels/$kernel.cubin").getPath
    val module = new CUmodule()
    var fxn = new CUfunction()

    cuModuleLoad(module, kernelPath)
    // We assume the name of the file is the name of the kernel
    cuModuleGetFunction(fxn, module, kernel)
    fxn
  }

  private def getImages = {
    val filePath = getClass.getResource("/mnist/train-images.idx3-ubyte").getPath
    val fis = new FileInputStream(filePath)
    var byte = new Array[Byte](4)
    fis.read(byte)
    println((ByteBuffer.wrap(byte).getInt()))
    fis.read(byte)
    println((ByteBuffer.wrap(byte).getInt()))
    fis.read(byte)
    var rows = ByteBuffer.wrap(byte).getInt()
    println("Number of rows: " + rows)
    fis.read(byte)
    var cols = ByteBuffer.wrap(byte).getInt()
    println("Number of cols: " + cols)

    var imageBytes = new Array[Byte](rows * cols)
    // Normally we want to get our labels here
    // but fuck it.
    fis.read(imageBytes)
    fis.read(imageBytes)
    fis.read(imageBytes)
    fis.read(imageBytes)
    // Close out our inputstream we got the goods
    fis.close()
    imageBytes.map(_.asInstanceOf[Float])
//    for(j <- 1 to 6000) {
//
//
//      fis.read(imageBytes)
//      val c = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY)
//      var i = 0
//      for ( byte <- imageBytes ) {
//        val y = i / 28
//        val x = i % 28
//        c.setRGB(x, y, byte)
//        i += 1
//      }
//
//      ImageIO.write(c, "jpg", new File(s"/home/dev/Numbers/${j}number.jpg"))
//    }
  }

  // Generates all the necessary random weights for our array
  def generateRandomWeightArray(layers: Int, size: Int) = {
    JCurand.setExceptionsEnabled(true)
    // We now need to build out our random
    // weight tensors
    val generator = new curandGenerator

    // We can parse this array into a scala readable object
    // but ehhh.
    val host = new Array[Float](size)

    // Create memory
    val devPtr = new CUdeviceptr
    cuMemAlloc(devPtr, size * Sizeof.FLOAT)

    curandCreateGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT)

    curandSetPseudoRandomGeneratorSeed(generator, Calendar.getInstance.getTimeInMillis)

    // Generate n floats on device
    curandGenerateUniform(generator, devPtr, size)
    cuCtxSynchronize

    // Copy device memory to host
    cuMemcpyDtoH(
      Pointer.to(host),
      devPtr,
      size * Sizeof.FLOAT
    )

    // Cleanup
    curandDestroyGenerator(generator)
    cuMemFree(devPtr)

    host
  }

  def NeuralNetwork1 = {
    val image = getImages

    JCudaDriver.setExceptionsEnabled(true)

    cuInit(0)

    val ctx = bindCtx
    val kernel = getKernel("NeuralNetwork1")

    //JCuda.cudaDeviceSetLimit(cudaLimit.cudaLimitStackSize, )
    val stackSize = new Array[Long](1)
    val heapSize = new Array[Long](1)
    JCuda.cudaDeviceGetLimit(stackSize, cudaLimit.cudaLimitStackSize)
    println("Our stack limit is: " + stackSize(0))
    JCuda.cudaDeviceGetLimit(heapSize, cudaLimit.cudaLimitMallocHeapSize)
    println("Our heap limit is: " + heapSize(0))

    val layers = 8
    val size = Math.pow(28, 4).asInstanceOf[Int] * layers

    println("Number of weights our NN contains: " + size)

    println("Readjusting our heap size to size of our neural net")
    val newHeapSize = heapSize(0) + size * Sizeof.FLOAT
    JCuda.cudaDeviceSetLimit(cudaLimit.cudaLimitMallocHeapSize, newHeapSize)

    println("New heap size is: " + newHeapSize)

    // Create a random normalized uniformed floating weight array
    var weightArray = generateRandomWeightArray(layers, size)

    val weights = new CUdeviceptr
    val l0 = new CUdeviceptr
    val results = new CUdeviceptr
    val softOuts = 10

    // Now we want to allocate memory for our weights
    cuMemAlloc(weights, size * Sizeof.FLOAT)
    cuMemcpyHtoD(
      weights,
      Pointer.to(weightArray),
      size * Sizeof.FLOAT
    )

    // Now we want to allocate memory for our initial layer
    cuMemAlloc(l0, size * Sizeof.FLOAT)
    cuMemcpyHtoD(
      l0,
      Pointer.to(image),
      size * Sizeof.FLOAT
    )

    // Lastly we want to allocate our results array
    cuMemAlloc(results, softOuts * Sizeof.FLOAT)

    // Now we want to allocate our kernel parameters
    val kernelParameters = Pointer.to(
      Pointer.to(weights),
      Pointer.to(l0),
      Pointer.to(Array[Int](28)),
      Pointer.to(Array[Int](28)),
      Pointer.to(Array[Int](size)),
      Pointer.to(Array[Int](layers)),
      Pointer.to(Array[Int](softOuts)),
      Pointer.to(results)
    )

    // Call the kernel function
    val blockSizeX = 16
    val blockSizeY = 16
    cuLaunchKernel(
      kernel,
      1, 1, 1,
      blockSizeX, blockSizeY, 1,
      0, null,
      kernelParameters, null
    )

    // Synchronize our context
    cuCtxSynchronize

    // Extract our precious matrix for our end result
    var resultsH = new Array[Float](10)
    cuMemcpyDtoH(
      Pointer.to(resultsH),
      results,
      Sizeof.FLOAT * softOuts
    )

    val used = new Array[Long](1)
    val free = new Array[Long](1)
    cuMemGetInfo(
      used,
      free
    )

    println("Used: " + used(0))
    println("Free: " + free(0))

    // Don't forget to free up all that memory
    cuMemFree(weights)
    cuMemFree(l0)
    cuMemFree(results)

    cuCtxDestroy(ctx)

    resultsH
  }

  def NeuralNetTrain = {
    // For now this will return one image to work with
    val image = getImages

    // We now have the image and now we want to setup cuda
    JCudaDriver.setExceptionsEnabled(true)

    cuInit(0)

    val ctx = bindCtx
    val kernel = getKernel("PreSum")

    val size = 28 * 28

    // Now we allocate device input
    val matrixD = new CUdeviceptr
    cuMemAlloc(matrixD, size * Sizeof.FLOAT)
    cuMemcpyHtoD(
      matrixD,
      Pointer.to(image),
      size * Sizeof.FLOAT
    )

    val weights = Array.fill[Float](28 * 28)(2)

    val weightsD = new CUdeviceptr
    cuMemAlloc(weightsD, size * Sizeof.FLOAT)
    cuMemcpyHtoD(
      weightsD,
      Pointer.to(weights),
      size * Sizeof.FLOAT
    )

    val accD = new CUdeviceptr
    cuMemAlloc(accD, Sizeof.FLOAT)

    val kernelParameters = Pointer.to(
      Pointer.to(matrixD),
      Pointer.to(weightsD),
      Pointer.to(Array[Int](28)),
      Pointer.to(Array[Int](28)),
      Pointer.to(Array[Int](size)),
      Pointer.to(accD)
    )

    // Call the kernel function
    val blockSizeX = 16
    val blockSizeY = 16
    cuLaunchKernel(
      kernel,
      1, 1, 1,
      blockSizeX, blockSizeY, 1,
      0, null,
      kernelParameters, null
    )

    // Synchronize our context
    cuCtxSynchronize

    // Extract our precious matrix for our end result
    var result = Array[Float](0)
    cuMemcpyDtoH(
      Pointer.to(result),
      accD,
      Sizeof.FLOAT
    )

    // Don't forget to free up all that memory
    cuMemFree(accD)
    cuMemFree(matrixD)

    // Clean up our context
    cuCtxDestroy(ctx)

    // Return our end result
    result
  }

  def JCudaVectorAddKernel = {
    JCudaDriver.setExceptionsEnabled(true)

    cuInit(0)

    val ctx = bindCtx
    val kernel = getKernel("JCudaVectorAddKernel")

    val numElements = 15000

    val hostInputA = 1 to numElements toArray
    val hostInputB = 1 to numElements toArray
    val SI: Int = Sizeof.INT

    // Allocate the device input data, and copy
    // the host input data to the device
    var deviceInputA = new CUdeviceptr
    cuMemAlloc(deviceInputA, numElements * SI)
    cuMemcpyHtoD(
      deviceInputA,
      Pointer.to(hostInputA),
      numElements * SI
    )

    var deviceInputB = new CUdeviceptr
    cuMemAlloc(deviceInputB, numElements * SI)
    cuMemcpyHtoD(
      deviceInputB,
      Pointer.to(hostInputB),
      numElements * SI
    )

    // Allocate device output memory
    val deviceOutput = new CUdeviceptr()
    cuMemAlloc(deviceOutput, numElements * SI)

    // Set up the kernel parameters: A pointer to an array
    // of pointers which point to the actual values.
    val kernelParameters = Pointer.to(
      Pointer.to(Array[Int](numElements)),
      Pointer.to(deviceInputA),
      Pointer.to(deviceInputB),
      Pointer.to(deviceOutput)
    )

    // Call the kernel function
    val blockSizeX = 256
    val gridSizeX = Math.ceil(numElements / blockSizeX).asInstanceOf[Int]
    cuLaunchKernel(
      kernel,
      gridSizeX, 1, 1,
      blockSizeX, 1, 1,
      0, null,
      kernelParameters, null
    )

    cuCtxSynchronize

    // Allocate host output memory and copy the device output
    // to the host
    var hostOutput = new Array[Int](numElements)
    cuMemcpyDtoH(
      Pointer.to(hostOutput),
      deviceOutput,
      numElements * SI
    )

    cuMemFree(deviceInputA)
    cuMemFree(deviceInputB)
    cuMemFree(deviceOutput)

    // Get rid of this so we don't have
    cuCtxDestroy(ctx)

    hostOutput
  }

  def JCudaMatrixSharedMemKernel(matrixA: Matrix, matrixB: Matrix) = {
    // Build host matrix that we will hold our values in
    val matrix = new Matrix(
      matrixB.width,
      matrixA.height,
      new Array[Float](matrixB.width * matrixA.height)
    )

    JCudaDriver.setExceptionsEnabled(true)

    cuInit(0)

    // Get device and context
    val ctx = bindCtx

    // Get kernel to run program
    val kernel = getKernel("JCudaMatrixSharedMemKernel")

    // Allocate device memory for matrix A
    val matrixA_D = new CUdeviceptr
    cuMemAlloc(matrixA_D, matrixA.size * Sizeof.FLOAT)
    cuMemcpyHtoD(
      matrixA_D,
      Pointer.to(matrixA.elements),
      matrixA.size * Sizeof.FLOAT
    )

    // Allocate device memory for matrix B
    val matrixB_D = new CUdeviceptr
    cuMemAlloc(matrixB_D, matrixB.size * Sizeof.FLOAT)
    cuMemcpyHtoD(
      matrixB_D,
      Pointer.to(matrixB.elements),
      matrixB.size * Sizeof.FLOAT
    )

    // Allocate device memory for matrix B
    val matrix_D = new CUdeviceptr
    // We don't care whats in matrix_D elements as this
    // becomes junk
    println("Size of: " + matrix.size * Sizeof.FLOAT)
    cuMemAlloc(matrix_D, matrix.size * Sizeof.FLOAT)

    // Setup our kernel parameters
    val kernelParameters = Pointer.to(
      Pointer.to(Array[Int](matrixA.width)),
      Pointer.to(Array[Int](matrixA.height)),
      Pointer.to(matrixA_D),
      Pointer.to(Array[Int](matrixB.width)),
      Pointer.to(Array[Int](matrixB.height)),
      Pointer.to(matrixB_D),
      Pointer.to(matrix_D)
    )

    // Using block size 16 X 16
    val gridX = matrix.height.asInstanceOf[Double] / 16
    val gridY = matrix.width.asInstanceOf[Double] / 16

    // Launch our kernel
    cuLaunchKernel(
      kernel,
      gridX.ceil.asInstanceOf[Int], gridY.ceil.asInstanceOf[Int], 1,
      16, 16, 1,
      0, null,
      kernelParameters, null
    )

    // Synchronize our context
    cuCtxSynchronize

    // Extract our precious matrix for our end result
    cuMemcpyDtoH(
      Pointer.to(matrix.elements),
      matrix_D,
      matrix.size * Sizeof.FLOAT
    )

    // Don't forget to free up all that memory
    cuMemFree(matrix_D)
    cuMemFree(matrixA_D)
    cuMemFree(matrixB_D)

    // Clean up our context
    cuCtxDestroy(ctx)

    // Return our end result
    matrix
  }

}

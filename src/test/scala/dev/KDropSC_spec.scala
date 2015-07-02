package dev

import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

import dev.TestCuda
import dev.la.Matrix
import org.scalatest.{FlatSpec, Matchers}
import scala.io.Source
import dev.KDropSC

/**
 * Created by dev on 5/18/15.
 */
class KDropSC_spec extends FlatSpec with Matchers {
  "readKernels" should "read my config file" in {
    val source = getClass.getResource("/config.json")
    val test = KDropSC.readKernels(source.getFile())

    // Do we have the correct keys?
    assert(test.contains("kernels"))
    assert(test.contains("kernelHelpers"))
    assert(test.size == 2)

    // Do we have all of the necessary kernels?
    val kernels = test.get("kernels").get
    assert(kernels.contains("test.kernel.cl"))

    // Does our source match?
    val sourceCode = kernels.get("test.kernel.cl").get
    val expected = "//Kernel 1"
    assert(sourceCode.substring(0,expected.length) == expected)

    // Do we have all of the necessary kernel helpers?
    val helpers = test.get("kernelHelpers").get
    assert(helpers.contains("exception.cl"))
    assert(helpers.contains("test.kernel.helper.cl"))

    // Does our source match?
    val sourceCodeException = helpers.get("exception.cl").get
    val sourceCodeTest = helpers.get("test.kernel.helper.cl").get
    val expectedException = "// Exception.cl"
    val expectedTest = "// Kernel Helper 1"

    assert(sourceCodeException.substring(0,expectedException.length) == expectedException)
    assert(sourceCodeTest.substring(0,expectedTest.length) == expectedTest)
  }

  ignore should "get platform and devices" in {
    val initialize = KDropSC.initializePlatformAndDevices()

    // This will tell us if we were successful in initialization
    assert(initialize == true)

    // Did we actually get a platform?
    val platform = KDropSC.getPlatform()
    assert(platform.isDefined)

    // Did we actually get a GPU?
    val devices = KDropSC.getDevices()
    assert(devices.isDefined)
  }

  it should "JCudaVectorAddKernel" in {
    TestCuda.JCudaVectorAddKernel
    assert(true)
  }

  it should "JCudaMatrixSharedMemKernel" in {
    val aF = new Array[Float](36)
    (0 to 35).map(_.asInstanceOf[Float]).copyToArray(aF)
    val bF = new Array[Float](36)
    (0 to 35).map(_.asInstanceOf[Float]).copyToArray(bF)

    val a = new Matrix(6,6, aF)
    val b = new Matrix(6,6, bF)
    val c = TestCuda.JCudaMatrixSharedMemKernel(a, b)
    assert(true)
  }

  it should "TestBlockSync" in {
    val output = TestCuda.TestBlockSync
    println("*****************")
    println("Testing block sync")
    output.foreach(println(_));
    println("*****************")
  }

  it should "PreSum" in {
    val output = TestCuda.PreSum(1)
    val output2 = TestCuda.PreSum(2)
    println("Our output: " + output(0))
    assert(output(0)*2 == output2(0))
  }

  it should "Neural Net" in {
    val results = TestCuda.QuantTensor
    assert(true)
  }

  it should "RunNNSimple" in {
    val result = TestCuda.RunNNSimple
    assert(true)
  }
}

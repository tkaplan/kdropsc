package dev

import java.io.File
import java.util.concurrent.Future

import jdk.nashorn.internal.parser.JSONParser
import org.jocl._
import org.jocl.CL._
import org.json.{JSONArray, JSONObject}
import scala.Predef
import scala.collection.JavaConversions._
import scala.collection.Map
import scala.io.Source
import java.io.File

import scala.util.matching.Regex

/**
 * Created by dev on 5/15/15.
 */
object KDropSC {

  def regexMappedFiles (filePath: String): Map[String, String] = {
    // If file is a directory then list all files
    // and return future map
    var mappedFiles = Map.empty[String, String]
    if (!filePath.contains("*")) {
      // Check if this is a relative path
      val file = new File(filePath)
      return Map(file.getName() -> Source.fromFile(filePath).getLines.mkString)
    }

    // If file name contains *, rip file into 3 parts
    // which filter with
    val filePathArray = filePath.split("/")
    val fileRegex  = filePathArray.last.split("\\*")

    var matchFirst: Regex = null
    var matchLast: Regex = null
    var allowAll: Boolean = false

    if (fileRegex.length > 1) {
      matchFirst = s"^${fileRegex(0)}.*".r
      matchLast  = s".*${fileRegex(1)}$$".r
    } else {
      allowAll = true
    }

    // Rip absolute path
    val path = filePathArray.slice(0,filePathArray.length - 1).mkString("/")

    // Rip file name
    val files = (new File(path)).list

    files.foreach {
      fileName =>
        val file = s"$path/$fileName"
        if (allowAll) {
          mappedFiles += (fileName -> Source.fromFile(file).getLines.mkString)
        } else {
          if (
            (matchFirst.pattern.matcher(fileName).matches) &&
            (matchLast.pattern.matcher(fileName).matches)
          ) {
            mappedFiles += (fileName -> Source.fromFile(file).getLines.mkString)
          }
        }
    }

    return mappedFiles
  }

  def getAbsPath(path: String, configAbsPath: String): String = {
    path.charAt(0) match {
      // Absolute path
      case '/' =>
        path
      // Relative path
      case _ =>
        configAbsPath + "/" + path
    }
  }

  // Get mapped files returns a map with the schema
  // ("FileName","FileStringContents")
  def getMappedFiles(array: JSONArray, absPath: String) = {
    var mappedFiles = Map.empty[String, String]
    val length = array.length()
    for (index <- 0 until length) {
      val item = array.getString(index)
      println(s"Getting directory/file for: ${getAbsPath(item, absPath)}")
      mappedFiles ++= regexMappedFiles(getAbsPath(item, absPath))
    }
    mappedFiles
  }

  def readConfigFile(filePath: String): Predef.Map[String, Map[String, String]] = {
    var config = Map.empty[String, Map[String, String]]

    // Read in file
    val file = new File(filePath)
    val source = Source.fromFile(filePath)
    val configStr = source.getLines.mkString
    val json = new JSONObject(configStr)
    val kernels = json.getJSONArray("kernels")
    val kernelHelpers = json.getJSONArray("kernelHelpers")

    config += ("kernels" -> getMappedFiles(kernels, file.getParent))
    config += ("kernelHelpers" -> getMappedFiles(kernelHelpers, file.getParent))

    return config
  }

  def initialize(config: String): Unit = {
    // Read from config file file
    // Find platform -1

    // Create contexts and command queues -1

    // Read config file to load kernel strings
    // and dependent headers -1

    // Stream kernel strings to build programs
    // and memory map :Dependent on -1

    // Build programs against contexts and devices
  }
}
//
///**
// * Created by dev on 4/16/15.
// */
//public class Hello {
//  /**
//   * The source code of the OpenCL program to execute
//   */
//  private static String programSource =
//    "__kernel void "+
//      "sampleKernel(__global const float *a,"+
//      "             __global const float *b,"+
//      "             __global float *c)"+
//      "{"+
//      "    int gid = get_global_id(0);"+
//      "    c[gid] = a[gid] * b[gid];"+
//      "}";
//
//  /**
//   * The entry point of this sample
//   *
//   * @param args Not used
//   */
//  public static void main(String args[])
//  {
//    // Create input- and output data
//    int n = 10;
//    float srcArrayA[] = new float[n];
//    float srcArrayB[] = new float[n];
//    float dstArray[] = new float[n];
//    for (int i=0; i<n; i++)
//    {
//      srcArrayA[i] = i;
//      srcArrayB[i] = i;
//    }
//    Pointer srcA = Pointer.to(srcArrayA);
//    Pointer srcB = Pointer.to(srcArrayB);
//    Pointer dst = Pointer.to(dstArray);
//
//    // The platform, device type and device number
//    // that will be used
//    final int platformIndex = 0;
//    final long deviceType = CL_DEVICE_TYPE_ALL;
//    final int deviceIndex = 0;
//
//    // Enable exceptions and subsequently omit error checks in this sample
//    CL.setExceptionsEnabled(true);
//
//    // Obtain the number of platforms
//    int numPlatformsArray[] = new int[1];
//    clGetPlatformIDs(0, null, numPlatformsArray);
//    int numPlatforms = numPlatformsArray[0];
//
//    // Obtain a platform ID
//    cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
//    clGetPlatformIDs(platforms.length, platforms, null);
//    cl_platform_id platform = platforms[platformIndex];
//
//    // Initialize the context properties
//    cl_context_properties contextProperties = new cl_context_properties();
//    contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
//
//    // Obtain the number of devices for the platform
//    int numDevicesArray[] = new int[1];
//    clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
//    int numDevices = numDevicesArray[0];
//
//    // Obtain a device ID
//    cl_device_id devices[] = new cl_device_id[numDevices];
//    clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
//    cl_device_id device = devices[deviceIndex];
//
//    // Create a context for the selected device
//    cl_context context = clCreateContext(
//      contextProperties, 1, new cl_device_id[]{device},
//      null, null, null);
//
//    // Create a command-queue for the selected device
//    cl_command_queue commandQueue =
//      clCreateCommandQueue(context, device, 0, null);
//
//    // Allocate the memory objects for the input- and output data
//    cl_mem memObjects[] = new cl_mem[3];
//    memObjects[0] = clCreateBuffer(context,
//    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
//    Sizeof.cl_float * n, srcA, null);
//    memObjects[1] = clCreateBuffer(context,
//    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
//    Sizeof.cl_float * n, srcB, null);
//    memObjects[2] = clCreateBuffer(context,
//    CL_MEM_READ_WRITE,
//    Sizeof.cl_float * n, null, null);
//
//    // Create the program from the source code
//    cl_program program = clCreateProgramWithSource(context,
//      1, new String[]{ programSource }, null, null);
//
//    // Build the program
//    clBuildProgram(program, 0, null, null, null, null);
//
//    // Create the kernel
//    cl_kernel kernel = clCreateKernel(program, "sampleKernel", null);
//
//    // Set the arguments for the kernel
//    clSetKernelArg(kernel, 0,
//      Sizeof.cl_mem, Pointer.to(memObjects[0]));
//    clSetKernelArg(kernel, 1,
//      Sizeof.cl_mem, Pointer.to(memObjects[1]));
//    clSetKernelArg(kernel, 2,
//      Sizeof.cl_mem, Pointer.to(memObjects[2]));
//
//    // Set the work-item dimensions
//    long global_work_size[] = new long[]{n};
//    long local_work_size[] = new long[]{1};
//
//    // Execute the kernel
//    clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
//      global_work_size, local_work_size, 0, null, null);
//
//    // Read the output data
//    clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0,
//    n * Sizeof.cl_float, dst, 0, null, null);
//
//    // Release kernel, program, and memory objects
//    clReleaseMemObject(memObjects[0]);
//    clReleaseMemObject(memObjects[1]);
//    clReleaseMemObject(memObjects[2]);
//    clReleaseKernel(kernel);
//    clReleaseProgram(program);
//    clReleaseCommandQueue(commandQueue);
//    clReleaseContext(context);
//
//    // Verify the result
//    boolean passed = true;
//    final float epsilon = 1e-7f;
//    for (int i=0; i<n; i++)
//    {
//      float x = dstArray[i];
//      float y = srcArrayA[i] * srcArrayB[i];
//      boolean epsilonEqual = Math.abs(x - y) <= epsilon * Math.abs(x);
//      if (!epsilonEqual)
//      {
//        passed = false;
//        break;
//      }
//    }
//    System.out.println("Test "+(passed?"PASSED":"FAILED"));
//    if (n <= 10)
//    {
//      System.out.println("Result: "+java.util.Arrays.toString(dstArray));
//    }
//  }
//}
//
//

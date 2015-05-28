package dev

import org.scalatest.{FlatSpec, Matchers}
import scala.io.Source
import dev.KDropSC

/**
 * Created by dev on 5/18/15.
 */
class KDropSC_spec extends FlatSpec with Matchers {
  "readConfig" should "read my json" in {
    val source = getClass.getResource("/config.json")
    val test = KDropSC.readConfigFile(source.getFile())

    // Do we have the correct keys?
    assert(test.contains("kernels"))
    assert(test.contains("kernelHelpers"))
    assert(test.size == 2)

    // Do we have all of the necessary kernels?

    // Do we have all of the necessary kernel helpers?
  }
}

package dev.la

/**
 * Created by dev on 6/15/15.
 */
sealed trait MatrixOps {
  val width: Int
  val height: Int
  def size = {
    width * height
  }
}

case class Matrix(width: Int, height: Int, elements: Array[Float]) extends MatrixOps

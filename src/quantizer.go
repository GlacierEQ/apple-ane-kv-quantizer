// Apple Neural Engine (ANE) FP16 Weight Quantizer
package main

import (
	"fmt"
	"math"
)

type QuantizedTensor struct {
	Scale  float32
	ZeroPoint int32
	Data   []int8
}

func QuantizeFloat32ToINT8(weights []float32) QuantizedTensor {
	var minVal, maxVal float32 = weights[0], weights[0]
	for _, w := range weights {
		if w < minVal { minVal = w }
		if w > maxVal { maxVal = w }
	}

	scale := (maxVal - minVal) / 255.0
	zeroPoint := int32(math.Round(float64(-minVal / scale)))

	qData := make([]int8, len(weights))
	for i, w := range weights {
		qVal := int32(math.Round(float64(w/scale))) + zeroPoint
		if qVal < -128 { qVal = -128 }
		if qVal > 127 { qVal = 127 }
		qData[i] = int8(qVal)
	}

	return QuantizedTensor{
		Scale: scale,
		ZeroPoint: zeroPoint,
		Data: qData,
	}
}

func main() {
	weights := []float32{0.12, -0.85, 0.44, 1.25, -1.10}
	tensor := QuantizeFloat32ToINT8(weights)
	fmt.Printf("ANE Quantized Tensor: Scale=%.4f, ZeroPoint=%d, Elements=%d\n", tensor.Scale, tensor.ZeroPoint, len(tensor.Data))
}

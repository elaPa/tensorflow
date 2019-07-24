/*
Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tensorflow

import (
	"fmt"
	"strconv"
	"strings"
	"testing"

	"github.com/golang/protobuf/proto"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	pb "github.com/tensorflow/tensorflow/core/protos_all_go_proto"
)

func TestSavedModel(t *testing.T) {
	bundle, err := LoadSavedModel("../cc/saved_model/testdata/half_plus_two/00000123", []string{"serve"}, nil)
	if err != nil {
		t.Fatalf("LoadSavedModel(): %v", err)
	}
	if op := bundle.Graph.Operation("y"); op == nil {
		t.Fatal("\"y\" not found in graph")
	}

	inputData := []string{}
	for _, val := range []float32{0, 1, 2, 3} {
		ex, err := exampleAsString(val)
		if err != nil {
			t.Fatalf("exampleAsString(%f) got err %v; want nil err", val, err)
		}
		inputData = append(inputData, ex)
	}

	inputTensor, err := NewTensor(inputData)
	if err != nil {
		t.Fatalf("NewTensor(%v) got err %v; want nil err", inputData, err)
	}

	signature := bundle.MetaGraphDef.GetSignatureDef()["regress_x_to_y"]
	input := signature.GetInputs()["inputs"].GetName()
	inName, inID, err := nameAndID(input)
	if err != nil {
		t.Fatalf("nameAndID(%v) got err %v; want nil err", input, err)
	}
	feeds := map[Output]*Tensor{
		bundle.Graph.Operation(inName).Output(inID): inputTensor,
	}

	output := signature.GetOutputs()["outputs"].GetName()
	outName, outID, err := nameAndID(output)
	if err != nil {
		t.Fatalf("nameAndID(%v) got err %v; want nil err", output, err)
	}
	fetches := []Output{
		bundle.Graph.Operation(outName).Output(outID),
	}

	results, err := bundle.Session.Run(feeds, fetches, nil)
	if err != nil {
		t.Fatalf("%v.Session.Run(%v, %v, nil) got err %v; want nil err", bundle, feeds, fetches, err)
	}

	if len(results) != 1 {
		t.Errorf("%v.Session.Run(%v, %v, nil) got results %v; want len 1", bundle, feeds, fetches, results)
	}

	wantVals := [][]float32{{2}, {2.5}, {3}, {3.5}}
	want, err := NewTensor(wantVals)
	if err != nil {
		t.Fatalf("NewTensor(%v) got err %v; want nil err", wantVals, err)
	}

	if diff := cmp.Diff(want, results[0], cmpopts.IgnoreUnexported(Tensor{})); diff != "" {
		t.Errorf("%v.Session.Run(%v, %v, nil) first result; -want +got:\n%s", diff)
	}
}

func nameAndID(opName string) (string, int, error) {
	s := strings.Split(opName, ":")
	if len(s) != 2 {
		return "", 0, fmt.Errorf("strings.Split(%v, \":\") got %v; want len 2", opName, s)
	}
	id, err := strconv.Atoi(s[1])
	return s[0], id, err

}

func exampleAsString(val float32) (string, error) {
	ex := &pb.Example{
		Features: &pb.Features{
			Feature: map[string]*pb.Feature{
				"x": &pb.Feature{
					Kind: &pb.Feature_FloatList{
						FloatList: &pb.FloatList{
							Value: []float32{val},
						},
					},
				},
			},
		},
	}

	b, err := proto.Marshal(ex)
	return string(b), err
}

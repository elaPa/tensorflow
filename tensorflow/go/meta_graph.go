/*
Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// #include "third_party/tensorflow/c/c_api.h"
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/golang/protobuf/proto"
	pb "github.com/tensorflow/tensorflow/tensorflow/core/protos_all_go_proto"
)

type metaGraph struct {
	c *C.TF_Buffer
}

func newMetaGraph() *metaGraph {
	return &metaGraph{C.TF_NewBuffer()}
}

func (mg *metaGraph) done() {
	C.TF_DeleteBuffer(mg.c)
}

func (mg *metaGraph) decode() (*pb.MetaGraphDef, error) {
	length := int(mg.c.length)
	if length > (1 << 30) {
		return nil, fmt.Errorf("MetaGraphDef is too large to decode, metaGraph.decode needs to be updated")
	}
	slice := (*[1 << 30]byte)(unsafe.Pointer(mg.c.data))[:length:length]
	mgpb := &pb.MetaGraphDef{}
	err := proto.Unmarshal(slice, mgpb)
	return mgpb, err
}

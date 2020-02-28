USE_MULTI_STREAM ?= 0

ifeq ($(USE_MULTI_STREAM), 1)
	COMMON_FLAGS += -DUSE_MULTI_STREAM=1
endif

GENCODE_FLAGS = -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70

google-net_cudnn:conv.o main.o activation.o pooling.o lrn.o concat.o dropout.o inception.o batch-inception.o loss.o softmax.o
	nvcc $^ -o $@ -lcudnn -lcublas ${GENCODE_FLAGS} $(COMMON_FLAGS)

main.o:main.cpp util.h conv.h activation.h pooling.h lrn.h
	nvcc -c $< -o $@ ${GENCODE_FLAGS} $(COMMON_FLAGS)
conv.o:conv.cpp util.h
	nvcc -c $< -o $@ ${GENCODE_FLAGS} $(COMMON_FLAGS)
activation.o:activation.cpp util.h
	nvcc -c $< -o $@ ${GENCODE_FLAGS} $(COMMON_FLAGS)
pooling.o:pooling.cpp util.h
	nvcc -c $< -o $@ ${GENCODE_FLAGS} $(COMMON_FLAGS)
lrn.o:lrn.cpp util.h
	nvcc -c $< -o $@ ${GENCODE_FLAGS} $(COMMON_FLAGS)
concat.o:concat.cu concat.h
	nvcc -c $< -o $@ ${GENCODE_FLAGS} $(COMMON_FLAGS)
dropout.o:dropout.cpp dropout.h
	nvcc -c $< -o $@ ${GENCODE_FLAGS} $(COMMON_FLAGS)
loss.o:loss.cpp loss.h
	nvcc -c $< -o $@ ${GENCODE_FLAGS} $(COMMON_FLAGS)
softmax.o:softmax.cpp softmax.h
	nvcc -c $< -o $@ ${GENCODE_FLAGS} $(COMMON_FLAGS)
inception.o:inception.cpp inception.h
	nvcc -c $< -o $@ ${GENCODE_FLAGS} $(COMMON_FLAGS)
batch-inception.o:batch-inception.cu batch-inception.h
	nvcc -c $< -o $@ ${GENCODE_FLAGS} $(COMMON_FLAGS)

clean:
	rm -f *.o

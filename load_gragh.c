/*************************************************************************
  > File Name: load_gragh.c
  > Author:perrynzhou 
  > Mail:perrynzhou@gmail.com 
  > Created Time: Sat 13 Apr 2019 12:06:29 AM CST
 ************************************************************************/

#include <stdio.h>
#include <tensorflow/c/c_api.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
static void releaseBuffer(void* data, size_t len)
{
    if (!data) {
        free(data);
        data = NULL;
    }
}
static TF_Buffer* initBufferFromFile(const char* file)
{
    int fd = -1;
    if (file == NULL || (fd = open(file, O_RDONLY)) == -1) {
        return NULL;
    }
    struct stat st;
    if (stat(file, &st) == -1) {
        close(fd);
        return NULL;
    }
    unsigned long sz = st.st_size;
    char* data = (char*)calloc(1, sz);
    if (data == NULL) {
        goto Err;
    }
    if (read(fd, data, sz) != sz) {
        goto Err;
    }
    TF_Buffer* tfb = TF_NewBuffer();
    tfb->data = (void*)data;
    tfb->length = sz;
    tfb->data_deallocator = releaseBuffer;
    close(fd);
    return tfb;
Err:
    if (fd != -1) {
        close(fd);
    }
    if (data != NULL) {
        free(data);
    }
    return NULL;
}
const char* tfDataTypeToString(TF_DataType data_type)
{
    switch (data_type) {
    case TF_FLOAT:
        return "TF_FLOAT";
    case TF_DOUBLE:
        return "TF_DOUBLE";
    case TF_INT32:
        return "TF_INT32";
    case TF_UINT8:
        return "TF_UINT8";
    case TF_INT16:
        return "TF_INT16";
    case TF_INT8:
        return "TF_INT8";
    case TF_STRING:
        return "TF_STRING";
    case TF_COMPLEX64:
        return "TF_COMPLEX64";
    case TF_INT64:
        return "TF_INT64";
    case TF_BOOL:
        return "TF_BOOL";
    case TF_QINT8:
        return "TF_QINT8";
    case TF_QUINT8:
        return "TF_QUINT8";
    case TF_QINT32:
        return "TF_QINT32";
    case TF_BFLOAT16:
        return "TF_BFLOAT16";
    case TF_QINT16:
        return "TF_QINT16";
    case TF_QUINT16:
        return "TF_QUINT16";
    case TF_UINT16:
        return "TF_UINT16";
    case TF_COMPLEX128:
        return "TF_COMPLEX128";
    case TF_HALF:
        return "TF_HALF";
    case TF_RESOURCE:
        return "TF_RESOURCE";
    case TF_VARIANT:
        return "TF_VARIANT";
    case TF_UINT32:
        return "TF_UINT32";
    case TF_UINT64:
        return "TF_UINT64";
    default:
        return "Unknown";
    }
}
void printOpInputs(TF_Graph* tfg, TF_Operation* op)
{
    const int num_inputs = TF_OperationNumInputs(op);

    fprintf(stdout, "Number inputs: %d\n", num_inputs);

    for (int i = 0; i < num_inputs; ++i) {
        const TF_Input input = { op, i };
        const TF_DataType type = TF_OperationInputType(input);
        fprintf(stdout, "%d type : %s\n", i, tfDataTypeToString(type));
    }
}
void printOpOutputs(TF_Graph* graph, TF_Operation* op)
{
    TF_Status* status = TF_NewStatus();
    const int num_outputs = TF_OperationNumOutputs(op);

    fprintf(stdout, "Number outputs: %d\n", num_outputs);

    for (int i = 0; i < num_outputs; ++i) {
        const TF_Output output = { op, i };
        const TF_DataType type = TF_OperationOutputType(output);
        fprintf(stdout, "%d type : %s ,", i, tfDataTypeToString(type));

        const int num_dims = TF_GraphGetTensorNumDims(graph, output, status);

        if (TF_GetCode(status) != TF_OK) {
            fprintf(stdout, "Can't get tensor dimensionality\n");
            continue;
        }

        fprintf(stdout, " dims: %d\n", num_dims);

        if (num_dims <= 0) {
            fprintf(stdout, " []\n");
            continue;
        }

        int64_t* dims = (int64_t*)malloc(sizeof(int64_t) * num_dims);
        TF_GraphGetTensorShape(graph, output, dims, num_dims, status);

        if (TF_GetCode(status) != TF_OK) {
            fprintf(stdout, "Can't get get tensor shape\n");
            continue;
        }
        char out[128] = { '\0' };
        sprintf(out, "%s", " [");
        for (int j = 0; j < num_dims; ++j) {
            sprintf(out + strlen(out), "%ld", dims[j]);
            if (j < num_dims - 1) {
                sprintf(out + strlen(out), "%s", ",");
            }
        }
        sprintf(out + strlen(out), "%s", "]");
        fprintf(stdout, "%s\n", out);
    }

    TF_DeleteStatus(status);
}
void printOp(TF_Graph* graph)
{
    TF_Operation* op;
    size_t pos = 0;

    while ((op = TF_GraphNextOperation(graph, &pos)) != NULL) {
        const char* name = TF_OperationName(op);
        const char* type = TF_OperationOpType(op);
        const char* device = TF_OperationDevice(op);

        const int num_outputs = TF_OperationNumOutputs(op);
        const int num_inputs = TF_OperationNumInputs(op);
        fprintf(stdout, "name:%s,type:%s,device:%s,num_outputs:%d,num_input:%d\n", name, type, device, num_inputs, num_outputs);
        printOpInputs(graph, op);
        printOpOutputs(graph, op);
    }
}
int main(int argc, char* argv[])
{
    if (argc != 2) {
        fprintf(stdout, "usage:%s %s\n", argv[0], "graph file");
        exit(0);
    }
    const char* file = argv[1];
    TF_Buffer* tfb = initBufferFromFile(file);
    if (tfb != NULL) {
        TF_Graph* graph = TF_NewGraph();
        TF_Status* status = TF_NewStatus();
        TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
        TF_GraphImportGraphDef(graph, tfb, opts, status);
        TF_DeleteImportGraphDefOptions(opts);
        printOp(graph);

        TF_DeleteBuffer(tfb);

        if (TF_GetCode(status) != TF_OK) {
            TF_DeleteStatus(status);
            TF_DeleteGraph(graph);
            fprintf(stdout, "import graph defination failed\n");
            return -1;
        }
        fprintf(stdout, "load graph success\n");
    }
    return 0;
}

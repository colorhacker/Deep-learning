#include "user_nn_debug.h"

#ifdef user_nn_debug_file


void user_nn_debug_printf(char *format,void *data) {
	static FILE *file_handle = NULL;
	file_handle = fopen(user_nn_debug_file, "a+");//×·¼Ó
	fprintf(file_handle, (const char *)format, data);
	fclose(file_handle);
}


#endif // user_nn_debug_file


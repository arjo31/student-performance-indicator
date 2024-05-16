import sys
import logging
import logger

def error_message_details(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()

    '''
     a tuple containing the type of the exception (a subclass of BaseException), the exception itself, and a traceback object which typically encapsulates the call stack at the point where the exception last occurred.
    '''
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_msg = "Error Occured in Python Script [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_msg

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail)

        # The constructor of the parent Exception class is called manually with the self.message argument using super().
    
    def __str__(self):
        return self.error_message

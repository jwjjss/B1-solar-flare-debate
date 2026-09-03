class MetadataError(Exception):
    def __init__(self, message:str):
        self.msg = message

    def __str__(self):
        return self.msg
    
class DuplicateError(Exception):
    def __init__(self, message:str):
        self.msg = message

    def __str__(self):
        return self.msg

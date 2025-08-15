DEBUG = True
from pathlib import Path
import os

LANGUAGES = ['ru', 'en', 'fr', 'de']

def debug_print(print_func):
    """Декоратор для создания отладочной версии функции печати"""
    def wrapper(*args, **kwargs):
        if DEBUG:
            # Добавляем префикс [DEBUG] к выводу
            new_args = ("[DEBUG]", *args)
            return print_func(*new_args, **kwargs)
        return None
    return wrapper

debug_print = debug_print(print)

#TODO: дупблирование определений классов. Вынести в отдельный модуль
class FileInfo:
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        
        if not self.filepath.exists():
            raise FileExistsError(f'file {self.filepath.absolute()} is not exists')
        
        self.base_name = "data_SMM4H_2025"
        self.parce_()
        
    def parce_extension_(self, filename):
        _, ext = os.path.splitext(filename)
        self.file_extension = ext

    def parce_filename_merkers_(self):
        complex_file_name = self.filename.replace(self.base_name, "")
        self.file_component_details = complex_file_name.split("_")
        if "" in self.file_component_details:
            self.file_component_details.remove("")

    def parce_lang_(self):
        self.lang = None    
        for lang in LANGUAGES:
            if lang in self.file_component_details:
                self.lang = lang

    def parce_(self):
        self.parce_extension_(self.filepath.name)
        self.filename = self.filepath.name.replace(self.file_extension, "")
        
        self.parce_filename_merkers_()
        
        self.parce_lang_()
        
        #TODO: потенциально можно выпилить
        # self.clean_data = False
        # if "clean" in self.file_component_details:
        #     self.clean_data = True
        
        # self.splited = False
        # if "splited" in self.file_component_details:
        #     self.splited = True
            
        # self.positive = None
        # if self.splited:
        #     if 'negative' in self.file_component_details:
        #         self.positive = False
        #     else:
        #         self.positive = True
                
        # self.json = False
        # if 'json' in self.file_component_details:
        #     self.json = True
            
        # self.generated = False
        # if 'generated' in self.file_component_details:
        #     self.generated = True
                
        self.data_root_dir = self.filepath.parent
        if self.splited or self.json or self.generated:
            self.data_root_dir = self.filepath.parent.parent

    @property
    def extension(self):
        return self.file_extension
    
    @property
    def lang(self):
        return self.lang
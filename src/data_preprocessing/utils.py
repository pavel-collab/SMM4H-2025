from pathlib import Path
import os

LANGUAGES = ['ru', 'en', 'fr', 'de']
class ParsedFileName:
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        
        if not self.filepath.exists():
            raise FileExistsError(f'file {self.filepath.absolute()} is not exists')
        
        self.base_name = "data_SMM4H_2025"
        
        self.parce_()
        
    def parce_(self):
        _, ext = os.path.splitext(self.filepath.name)
        self.file_extension = ext
        self.filename = self.filepath.name.replace(ext, "")
        
        complex_file_name = self.filename.replace(self.base_name, "")
        file_component_details = complex_file_name.split("_")
        if "" in file_component_details:
            file_component_details.remove("")
        
        self.lang = None    
        for lang in LANGUAGES:
            if lang in file_component_details:
                self.lang = lang
        
        self.clean_data = False
        if "clean" in file_component_details:
            self.clean_data = True
        
        self.splited = False
        if "splited" in file_component_details:
            self.splited = True
            
        self.positive = None
        if self.splited:
            if 'negative' in file_component_details:
                self.positive = False
            else:
                self.positive = True
                
        self.json = False
        if 'json' in file_component_details:
            self.json = True
            
        self.generated = False
        if 'generated' in file_component_details:
            self.generated = True
                
        self.data_root_dir = self.filepath.parent
        if self.splited or self.json or self.generated:
            self.data_root_dir = self.filepath.parent.parent
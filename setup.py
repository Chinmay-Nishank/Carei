from setuptools import setup,find_packages

with open("requirment.txt") as f :
    requir = f.read().splitlines()
    
setup(
    name = "medical_assistant",
    packages = find_packages(),
    install_requires= requir,
    
    
    
)    
    
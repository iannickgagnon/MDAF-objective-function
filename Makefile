# Makefile for generating UML diagrams using pyreverse and Graphviz (.dot)

PACKAGE_PATH=src/MDAF_benchmarks/
PROJECT_NAME=MDAF_benchmarks
OUTPUT_FORMAT=png

class-diagram:
	pyreverse -p $(PROJECT_NAME) $(PACKAGE_PATH)
	dot -T$(OUTPUT_FORMAT) classes_$(PROJECT_NAME).dot -o classes_$(PROJECT_NAME).$(OUTPUT_FORMAT)
	dot -T$(OUTPUT_FORMAT) packages_$(PROJECT_NAME).dot -o packages_$(PROJECT_NAME).$(OUTPUT_FORMAT)

class-diagram-plantuml:
	pyreverse -o plantuml -p $(PROJECT_NAME) $(PACKAGE_PATH)

#plantuml classes_$(PROJECT_NAME).plantuml
#plantuml packages_$(PROJECT_NAME).plantuml

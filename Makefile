# Makefile for generating UML diagrams using pyreverse and Graphviz (.dot)

PACKAGE_PATH=src/MDAF_objective_functions/
PROJECT_NAME=MDAF_objective_functions
OUTPUT_FORMAT=png

class-diagram:
	pyreverse -p $(PROJECT_NAME) $(PACKAGE_PATH)
	dot -T$(OUTPUT_FORMAT) classes_$(PROJECT_NAME).dot -o classes_$(PROJECT_NAME).$(OUTPUT_FORMAT)
	dot -T$(OUTPUT_FORMAT) packages_$(PROJECT_NAME).dot -o packages_$(PROJECT_NAME).$(OUTPUT_FORMAT)

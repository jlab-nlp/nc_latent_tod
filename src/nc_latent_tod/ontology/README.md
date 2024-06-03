# Ontology

This will hold ontologies for each  dialogue system we evaluate on, with the ideal of
moving towards a general-purpose ontology system. The goal is with an ontology, we can:

- recognize and normalize categorical slot values
- recognize special format values like times
- given a database of non-categorical entities, unify surface forms for the same kind of
entity (e.g. `"acorn guest house" == "the acorn guest house" == "acron guest house"`)

## MultiWOZ

Borrowing from our previous work (RefPyDST), we use the ontology system created there, named here
as `MultiWOZOntology` in `multiwoz/ontology.py`


The set of operations are designed to handle simple (mainly non-compositional) KB queries

* `relation(entity)`: a relation in KB that functions on an object and returns a set of subjects which satisfy the relation

* `reverse`: reverse the direction of a relation

* `count(set)`: returns the cardinality of the set

* `argmin/argmax(set, relation)`: returns the subject in the set where the object of the relation is min/max.
(a simpler non-compositional version used is `argmin/argmax(relation)`)

* `and(set1, set2)`: take the intersection of two sets

* `or(set1, set2)`: take the union of two sets

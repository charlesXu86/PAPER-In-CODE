The set of operations are designed to handle table queries

* `filter(table, condition)`: return the rows that satisfy a specified condition

* `op(column_name, entity)`: the condition used in filter, `op` is one of `== != > < >= <= include` 

* `count(table)`: return the number of rows

* `argmax/argmin(table, column_name)`: return the row where the value of a specified column is largest

* `size(column_name)`: wraps a column name, so that when used in `filter/argmax/argmin`, we use the cardinality of each entry for comparison


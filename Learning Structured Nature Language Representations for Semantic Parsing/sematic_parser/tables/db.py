from tinydb import TinyDB, Query, Table, TableExecutor
db = Table()
#db = TinyDB('db.json')
executor = TableExecutor()
User = Query()
db.insert({'name': 'John', 'age': 22, 'gender': 'm', 'ismarried': False, 'students': ['jack', 'jim']})
db.insert({'name': 'Julie', 'age': 23, 'gender': 'f', 'ismarried': False, 'students': ['joe']})
db.insert({'name': 'Jul', 'age': 22, 'gender': 'f', 'ismarried': False, 'students': ['jack', 'jim']})

#lf = 'count(argmax(filter(all >(size(column.students) 1)) column.age))'
#lf = 'display.name(filter(filter(all, include(column.students jack)) ==(column.gender f)))'
lf = 'count(argmax(filter(all >=(size(column.gender) 1)) age))'


k = executor.execute(lf, db)
print(k)
'''
k = db.search(User.students.count_ge(1))
print k

db2 = Table()
db2.insert_multiple(k)

k = db.search(User.students.count_gt(1))
print k


db3 = Table()
db3.insert_multiple(k)
#print db3.denotation_flatten_set('students')
print db3.argmax('age')
'''

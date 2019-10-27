
test_out_filename = "Ecommerce_test_out.txt"

with open(test_out_filename, 'r') as f:
	num_query = 0
	recall = {"recall@1": 0,
			  "recall@2": 0,
			  "recall@5": 0}

	lines = f.readlines()
	relevance_list = []
	for i,line in enumerate(lines[1:]):
		line = line.strip().split('\t')
		line = [float(ele) for ele in line]

		relevance_list.append(line[4])
		if (i+1) % 10 == 0:
			num_correct = sum(relevance_list)

			if num_correct == 0:
				relevance_list = []
			else:
				num_query += 1
				relevance_list = [ele/num_correct for ele in relevance_list]
				recall["recall@1"] += sum(relevance_list[:1])
				recall["recall@2"] += sum(relevance_list[:2])
				recall["recall@5"] += sum(relevance_list[:5])
				relevance_list = []

	recall["recall@1"] = recall["recall@1"] / float(num_query)
	recall["recall@2"] = recall["recall@2"] / float(num_query)
	recall["recall@5"] = recall["recall@5"] / float(num_query)
	print("num_query = {}".format(num_query))
	print("recall@1 = {}".format(recall["recall@1"]))
	print("recall@2 = {}".format(recall["recall@2"]))
	print("recall@5 = {}".format(recall["recall@5"]))



	

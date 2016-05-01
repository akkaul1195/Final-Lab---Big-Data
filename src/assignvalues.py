f = open("submission_test_file.csv.csv", 'r')
o = open("new_submissions.csv", 'w')

lines = f.readlines()
for line in lines:
	words = line.split(",")
	if ((words[1]) < .5):
		o.write(words[0] + "," + str(0))
	else:
		o.write(words[0] + "," + str(1))
f.close
o.close
print ("done")
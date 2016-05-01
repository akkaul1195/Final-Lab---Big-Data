f = open("submission_test_file.csv.csv", 'r')
o = open("new_submissions.csv", 'w')
o.write(f.readline())
lines = f.readlines()
for line in lines:
	words = line.split(",")
	if ((words[1]) < .5):
		o.write(words[0] + "," + str(0) + "\n")
	else:
		o.write(words[0] + "," + str(1) + "\n")
f.close
o.close
print ("done")
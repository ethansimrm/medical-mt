input = open("old_M2M100_enfr_FT_choi_2022_pred.txt", "r", encoding = "utf8")
output = open("M2M100_enfr_FT_choi_2022_pred.txt", "w", encoding = "utf8")
for line in input.readlines():
	line = line.replace("__fr__ ", "")
	output.write(line)
input.close()
output.close()

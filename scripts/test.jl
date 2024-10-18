begin
	using DataFrames
	using MLJ
	using DecisionTree
	using ConformalPrediction
	using TidierPlots
	using TidierFiles
	import MLJDecisionTreeInterface
	using Plots
end

# ╔═╡ 9b431fbb-a8b5-426c-aeae-c1c30a93aba7
begin
	path = "/Users/joepaul/Documents/PhD/Projects/Conformal-Prediction/src/data/cleaned/housing_clean.csv"
	df = read_csv(path)
end

# ╔═╡ af9b5726-8cb1-48eb-82a8-c1fd17e59115
schema(df)

# ╔═╡ c0e165e3-c078-439b-8479-b377a742dc98
begin	
	# Assuming df is your DataFrame
	X = select(df, Not(:SalePrice))  # Equivalent to df.drop("SalePrice", axis=1)
	# X = select(df, :GrLivArea)
	# X = df[:, :BedroomAbvGr] |> x -> reshape(x, :, 1)  # Reshape to mimic .to_numpy().reshape(-1,1)
	y = log.(df.SalePrice)  # Equivalent to np.log(df["SalePrice"])
	
	# Split the data into train, cal, and test indices
	n = nrow(df)
	train_ind, test_ind = partition(eachindex(y), 0.6; shuffle=true)
	test_ind, cal_ind =  partition(eachindex(test_ind), 0.5; shuffle=true)
end

# ╔═╡ 93b21974-7e08-407d-bc07-9cc889f698dd
begin
	random_forest = @load RandomForestRegressor pkg=DecisionTree verbosity=0
end

# ╔═╡ b5ef7486-0ddb-4d5f-a32a-f62c77749dea
println("Available models: ", collect(keys(merge(values(available_models[:regression])...))))

# ╔═╡ eb06ad7a-cf5e-41bb-8ff3-fba4b742107e
begin
	rf = random_forest()
	conf_model = conformal_model(rf; method=:simple_inductive)
	machine_raw = machine(conf_model, X, y)
	MLJ.fit!(machine_raw, rows=train_ind)
end



# ╔═╡ ebce2255-2391-4e33-befe-a397f8bd0363
begin
	Xtest = selectrows(X, test_ind)
	ytest = selectrows(y, test_ind)
end

# ╔═╡ 9d2be9b4-3b41-40c5-9282-e51620687130
ŷ = MLJ.predict(machine_raw, Xtest)

# ╔═╡ e1bb0a97-c205-43cb-b99f-3bd9b05cbbb1
begin
	_eval = evaluate!(
		machine_raw,
		operation=MLJ.predict,
		measure=[emp_coverage, ssc]
	)
	display(_eval)
	emp_cov = _eval.measurement[1]
	ssc_cov = _eval.measurement[2]
end

# ╔═╡ ac11d0c2-2fb6-4b5a-8c69-bb8f9b6676cd
begin
	# Extract lower and upper bounds from ŷ
	lower_bounds = [ci[1] for ci in ŷ]
	upper_bounds = [ci[2] for ci in ŷ]
	
	# Calculate the midpoints (mean predictions)
	predicted_mean = [(ci[1] + ci[2]) / 2 for ci in ŷ]
	
	# Calculate the error (distance from the midpoint to the bounds)
	yerr = [(ci[2] - (ci[1] + ci[2]) / 2, (ci[1] + ci[2]) / 2 - ci[1]) for ci in ŷ]
	
	# Plot the true values on the x-axis and the predicted mean with error bars on the y-axis
	scatter(ytest, predicted_mean, yerr=yerr, label="Predicted with Error Bars", legend=:bottomright)
	xlabel!("True Log Price")
	ylabel!("Predicted Log Price")
	annotate!(11.2, 13, "Coverage $(round(emp_cov, digits=3))")
	
	# Add a 45-degree line for reference
	plot!(x -> x, label="45-degree Line", color=:black, linestyle=:dash)

end

# ╔═╡ e51046a9-6045-43fe-bcdf-95e984d11d58
begin
	scatter(Xtest.GrLivArea, predicted_mean, yerr=yerr)
	scatter!(Xtest.GrLivArea, ytest)
	xlabel!("GrLivArea")
	ylabel!("Log House Price")
end


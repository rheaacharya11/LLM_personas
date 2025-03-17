def test_two_query_approach(model_path_prefix="../models/", model_name="llama3-8b-instruct"):
    """
    Test the two-query fairness assessment approach with a single comparison and basic persona.
    
    Args:
        model_path_prefix: Directory prefix for model path
        model_name: Name of the model to use
        
    Returns:
        None (prints results)
    """
    print("Testing two-query fairness assessment approach...")
    
    # Load COMPAS data
    train_df, _ = load_precleaned_data()
    
    # Generate a single comparison
    random.seed(42)
    individual1 = train_df.iloc[random.randint(0, len(train_df)-1)]
    individual2 = train_df.iloc[random.randint(0, len(train_df)-1)]
    
    # Create a simple persona
    persona = "a fair-minded judge with 20 years of experience in criminal justice"
    
    # Load model and tokenizer
    print(f"Loading {model_name} model...")
    model_path = f"{model_path_prefix}{model_name}"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_instance = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Get model enum
    model_enum = LlamaModel.LLAMA3_8B
    if "70b" in model_name:
        model_enum = LlamaModel.LLAMA3_70B
    
    # Print individual details
    print("\nPERSON X:")
    print(f"- Gender: {individual1['sex']}")
    print(f"- Age: {individual1['age']}")
    print(f"- Race: {individual1['race']}")
    print(f"- Juvenile felonies: {individual1['juv_fel_count']}")
    print(f"- Juvenile misdemeanors: {individual1['juv_misd_count']}")
    print(f"- Juvenile other: {individual1['juv_other_count']}")
    print(f"- Prior convictions: {individual1['priors_count']}")
    print(f"- Current charge severity: {individual1['c_charge_degree_full']}")
    
    print("\nPERSON Y:")
    print(f"- Gender: {individual2['sex']}")
    print(f"- Age: {individual2['age']}")
    print(f"- Race: {individual2['race']}")
    print(f"- Juvenile felonies: {individual2['juv_fel_count']}")
    print(f"- Juvenile misdemeanors: {individual2['juv_misd_count']}")
    print(f"- Juvenile other: {individual2['juv_other_count']}")
    print(f"- Prior convictions: {individual2['priors_count']}")
    print(f"- Current charge severity: {individual2['c_charge_degree_full']}")
    
    # Run the two-query approach
    print("\nRunning two-query approach...")
    response = fairness_query_two_step(
        comparison_id=0,
        individual1=individual1,
        individual2=individual2,
        model_instance=model_instance,
        tokenizer=tokenizer,
        persona=persona,
        model=model_enum
    )
    
    # Print results
    print("\nX vs Y QUERY RESPONSE:")
    print(response.x_vs_y_response)
    
    print("\nY vs X QUERY RESPONSE:")
    print(response.y_vs_x_response)
    
    x_vs_y_answer, x_vs_y_explanation = extract_yes_no_and_explanation(response.x_vs_y_response)
    y_vs_x_answer, y_vs_x_explanation = extract_yes_no_and_explanation(response.y_vs_x_response)
    
    print("\nEXTRACTED ANSWERS:")
    print(f"X vs Y: {x_vs_y_answer} - {x_vs_y_explanation}")
    print(f"Y vs X: {y_vs_x_answer} - {y_vs_x_explanation}")
    
    print("\nCOMBINED RESULT:")
    print(f"Choice Type: {response.choice}")
    print(f"Explanation: {response.explanation}")
    
    print("\nTest complete!")
    
    # Clean up to free memory
    del model_instance
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Run the test
    test_two_query_approach()
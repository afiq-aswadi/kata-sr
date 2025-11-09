use kata_sr::core::kata_loader::load_available_katas;

#[test]
fn test_load_real_katas() {
    let katas = load_available_katas().expect("should load katas");

    assert!(!katas.is_empty(), "should find at least one kata");

    for kata in &katas {
        assert!(!kata.name.is_empty(), "kata name should not be empty");
        // Category is optional if tags are provided (multi-tag support)
        assert!(
            !kata.category.is_empty() || !kata.tags.is_empty(),
            "kata '{}' should have either category or tags",
            kata.name
        );
        assert!(
            kata.base_difficulty >= 1 && kata.base_difficulty <= 5,
            "base_difficulty should be between 1 and 5, got {}",
            kata.base_difficulty
        );
        assert!(
            !kata.description.is_empty(),
            "kata description should not be empty"
        );
    }

    let kata_names: Vec<&str> = katas.iter().map(|k| k.name.as_str()).collect();
    println!("Loaded katas: {:?}", kata_names);

    assert!(kata_names.contains(&"mlp"), "should load mlp kata");
    assert!(
        kata_names.contains(&"multihead_attention"),
        "should load multihead_attention kata"
    );
}

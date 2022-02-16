#include "pmap.hh"

PrefixPermutationMap::PrefixPermutationMap()
    : pmap(new PrefixMap) {}

CapturedPermutationMap::CapturedPermutationMap()
    : pmap(new CapturedMap) {}

PrefixPermutationMap::~PrefixPermutationMap() {
    if(pmap)
        delete pmap;
}

CapturedPermutationMap::~CapturedPermutationMap() {
    if(pmap)
        delete pmap;
}

Node* PrefixPermutationMap::insert (unsigned short new_rule, size_t nrules, bool prediction, 
        bool default_prediction, double lower_bound, double objective, Node* parent, 
        int num_not_captured, int nsamples, int len_prefix, double c, double equivalent_minority,
        CacheTree* tree, VECTOR not_captured, tracking_vector<unsigned short, 
        DataStruct::Tree> parent_prefix, confusion_matrix_data* confusion_matrix) {
    (void) not_captured;
    logger->incPermMapInsertionNum();
    parent_prefix.push_back(new_rule);
    
    unsigned char* ordered = (unsigned char*) malloc(sizeof(unsigned char) * (len_prefix + 1));
    ordered[0] = (unsigned char)len_prefix;

    for (int i = 1; i < (len_prefix + 1); i++)
	    ordered[i] = i - 1;

    std::function<bool(int, int)> cmp = [&](int i, int j) { return parent_prefix[i] < parent_prefix[j]; };
    std::sort(&ordered[1], &ordered[len_prefix + 1], cmp);
    
    std::sort(parent_prefix.begin(), parent_prefix.end());
    // key now defines the prefix, but also its confusion matrix,
    // hence we consider equivalent prefixes as prefixes:
    // - implying the same rules (in possibly different orders)
    // - exhibiting the same confusion matrix
    unsigned short *pre_key = (unsigned short*) malloc(sizeof(unsigned short) * (len_prefix + 1 +8)); // modif : +8 to embbedd confusion matrix data
    pre_key[0] = (unsigned short)len_prefix;
    memcpy(&pre_key[1], &parent_prefix[0], len_prefix * sizeof(unsigned short));
    //printf("--\n");
    for(int index_in_tuple = 0;index_in_tuple < 8; index_in_tuple++){
        pre_key[len_prefix+1+index_in_tuple] = (*confusion_matrix)[index_in_tuple];
        //printf("%d\n",(int)(*confusion_matrix)[index_in_tuple]);
    }
    //printf("--\n");
    /*for(int i = 0; i < (len_prefix + 1 +8); i++){
        printf("%d\n",(int)pre_key[i]);
    }
    printf(" == \n");*/
    logger->addToMemory((len_prefix + 1) * (sizeof(unsigned char) + sizeof(unsigned short)), DataStruct::Pmap);

    prefix_key key = { pre_key };
    
    Node* child = NULL;
    PrefixMap::iterator iter = pmap->find(key);
    /*if (iter != pmap->end()) {
        double permuted_lower_bound = iter->second.first;
        if (lower_bound < permuted_lower_bound) { // <- this line was "if (lower_bound < permuted_lower_bound) {"
            Node* permuted_node;
            tracking_vector<unsigned short, DataStruct::Tree> permuted_prefix(parent_prefix.size());
            unsigned char* indices = iter->second.second;
            for (unsigned char i = 0; i < indices[0]; ++i)
                permuted_prefix[i] = parent_prefix[indices[i + 1]];
            if ((permuted_node = tree->check_prefix(permuted_prefix)) != NULL) {
                Node* permuted_parent = permuted_node->parent();
                permuted_parent->delete_child(permuted_node->id());
                delete_subtree(tree, permuted_node, false, tree->calculate_size());
                logger->incPmapDiscardNum();
            } else {
                logger->incPmapNullNum();
            }
            child = tree->construct_node(new_rule, nrules, prediction,
                                     default_prediction, lower_bound, objective,
                                     parent, num_not_captured, nsamples,
                                     len_prefix, c, equivalent_minority);
            iter->second = std::make_pair(lower_bound, ordered);
        }
    } else {
        child = tree->construct_node(new_rule, nrules, prediction,
                                 default_prediction, lower_bound, objective,
                                 parent, num_not_captured, nsamples, len_prefix,
                                 c, equivalent_minority);
        unsigned char* ordered_prefix = &ordered[0];
        pmap->insert(std::make_pair(key, std::make_pair(lower_bound, ordered_prefix)));
        logger->incPmapSize();
    }*/
    if (iter != pmap->end()) {
        for(int index_in_tuple = 0;index_in_tuple < len_prefix+1+8; index_in_tuple++){
            if((iter->first.key)[index_in_tuple] != key.key[index_in_tuple]){
                printf("Found key, but error, keys don't match!\n");
                for(int index_in_tuple_inner = 0;index_in_tuple_inner < len_prefix+1+8; index_in_tuple_inner++){
                    printf("map: %hu, prefix: %hu\n",  (iter->first.key)[index_in_tuple_inner],  key.key[index_in_tuple_inner]);
                }
                exit(-1);
            }
    } 
    } else{ // add node only if no other equivalent permutation already in queue

        child = tree->construct_node(new_rule, nrules, prediction,
                                 default_prediction, lower_bound, objective,
                                 parent, num_not_captured, nsamples, len_prefix,
                                 c, equivalent_minority);
        unsigned char* ordered_prefix = &ordered[0];
        pmap->insert(std::make_pair(key, std::make_pair(lower_bound, ordered_prefix)));
        logger->incPmapSize();

    
    }
    return child;
}

Node* CapturedPermutationMap::insert(unsigned short new_rule, size_t nrules, bool prediction, 
        bool default_prediction, double lower_bound, double objective, Node* parent, int num_not_captured, 
        int nsamples, int len_prefix, double c, double equivalent_minority, CacheTree* tree, 
        VECTOR not_captured, tracking_vector<unsigned short, DataStruct::Tree> parent_prefix, confusion_matrix_data* confusion_matrix) {
    logger->incPermMapInsertionNum();
    parent_prefix.push_back(new_rule);
    Node* child = NULL;
    captured_key key;
    rule_vinit(nsamples, &key.key);
    rule_copy(key.key, not_captured, nsamples);
#ifndef GMP
    key.len = (short) nsamples;
#endif
    CapturedMap::iterator iter = pmap->find(key);
    if (iter != pmap->end()) {
        double permuted_lower_bound = iter->second.first;
        tracking_vector<unsigned short, DataStruct::Tree> permuted_prefix = iter->second.second;
        if (lower_bound < permuted_lower_bound) {
            Node* permuted_node;
            if ((permuted_node = tree->check_prefix(permuted_prefix)) != NULL) {
                Node* permuted_parent = permuted_node->parent();
                permuted_parent->delete_child(permuted_node->id());
                delete_subtree(tree, permuted_node, false, tree->calculate_size());
                logger->incPmapDiscardNum();
            } else {
                logger->incPmapNullNum();
            }
            child = tree->construct_node(new_rule, nrules, prediction, default_prediction,
                                       lower_bound, objective, parent,
                                        num_not_captured, nsamples, len_prefix, c, equivalent_minority);
            iter->second = std::make_pair(lower_bound, parent_prefix);
        }
    } else {
        child = tree->construct_node(new_rule, nrules, prediction, default_prediction,
                                    lower_bound, objective, parent,
                                    num_not_captured, nsamples, len_prefix, c, equivalent_minority);
        pmap->insert(std::make_pair(key, std::make_pair(lower_bound, parent_prefix)));
        logger->incPmapSize();
    }
    return child;
}


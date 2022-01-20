#imports
import pathlib, shutil
from hashlib import sha512

def split_model_files(model_dir_path=pathlib.Path(__file__).parent/'nnunet_models',
                      bytes_per_chunk=50000000,remove=True) :
    """
    Recursively search a directory for any "model_final_checkpoint.model" files 
    and split them into smaller chunks so they can be uploaded as part of the GitHub repo

    model_dir_path = the top directory beneath which anything called "model_final_checkpoint.model" should be split
    bytes_per_chunk = the maximum number of bytes that should be included in each individual chunk
    remove = True if the original, single, large "model_final_checkpoint.model" files should be removed 
        once the split files are all created
    """
    if not model_dir_path.is_dir() :
        raise FileNotFoundError(f'ERROR: {model_dir_path} is not a directory!')
    for fp in model_dir_path.rglob('*') :
        if fp.name=='model_final_checkpoint.model' :
            ref_hash = sha512()
            with open(fp,'rb') as ifpo :
                ref_hash.update(ifpo.read())
            ref_hash = ref_hash.digest()
            new_hash_1 = sha512()
            created_fns = []
            with open(fp,'rb') as ifpo :
                chunk_i = 0
                new_bytes = ifpo.read(bytes_per_chunk)
                while len(new_bytes)>0 :
                    new_hash_1.update(new_bytes)
                    new_fn = f'{fp.name}_chunk_{chunk_i}'
                    created_fns.append(new_fn)
                    with open(fp.parent/new_fn,'wb') as ofpo :
                        ofpo.write(new_bytes)
                    chunk_i+=1
                    new_bytes = ifpo.read(bytes_per_chunk)
            new_hash_1 = new_hash_1.digest()
            new_hash_2 = sha512()
            if remove :
                for new_fn in created_fns :
                    if not (fp.parent/new_fn).is_file() :
                        remove = False
                        break
                    else :
                        with open(fp.parent/new_fn,'rb') as newfp :
                            new_hash_2.update(newfp.read())
            new_hash_2 = new_hash_2.digest()
            if not ref_hash==new_hash_1==new_hash_2 :
                errmsg = f'ERROR: hashes of original {fp} file and individual chunks are mismatched! '
                errmsg+= 'Original file will not be removed.'
                raise RuntimeError(errmsg)
            if remove :
                fp.unlink()

def assemble_model_files(model_dir_path=pathlib.Path(__file__).parent/'nnunet_models',remove=False) :
    """
    Recursively search a directory for any "fold_n" subdirectories and, if those subdirectories 
    contain any "model_final_checkpoint.model_chunk_n" files, assemble the individual files into 
    a single larger "model_final_checkpoint.model" file

    model_dir_path = the top directory beneath which any "fold_n" subdirectories should be searched for 
        "model_final_checkpoint.model_chunk_n" files to combine
    remove = True if the smaller individual files should be removed once they've been successfully rebuilt 
    """
    if not model_dir_path.is_dir() :
        raise FileNotFoundError(f'ERROR: {model_dir_path} is not a directory!')
    for fold_dir in model_dir_path.rglob('fold_*') :
        if not (fold_dir/'model_final_checkpoint.model_chunk_0').is_file() :
            continue
        newfp = fold_dir/'model_final_checkpoint.model'
        ref_hash = sha512()
        with open(newfp,'wb') as newf :
            for fp in sorted(fold_dir.glob('model_final_checkpoint.model_chunk_*')) :
                with open(fp,'rb') as oldf :
                    shutil.copyfileobj(oldf,newf)
                with open(fp,'rb') as oldf :
                    ref_hash.update(oldf.read())
        ref_hash = ref_hash.digest()
        new_hash = sha512()
        if newfp.is_file() :
            with open(newfp,'rb') as newf :
                new_hash.update(newf.read())
        new_hash = new_hash.digest()
        if not ref_hash==new_hash :
            errmsg = f'ERROR: hashes of re-assembled {newfp} file and individual chunks are mismatched! '
            errmsg+= 'Individual chunk files will not be removed.'
            raise RuntimeError(errmsg)
        if remove :
            for fp in fold_dir.glob('model_final_checkpoint.model_chunk_*') :
                fp.unlink()
